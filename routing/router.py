from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import CRS, Transformer
from loguru import logger
from shapely.geometry import LineString, mapping
from shapely.ops import transform

from classifier.archetype_classifier import ARCHETYPE_NAMES, predict_archetype
from rl.agents import PPOPolicyRouteAgent, QLearningRouteAgent, agent_path_for_archetype
from rl.env import NightRouteEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "scored_graph.graphml"
CLASSIFIER_PATH = PROJECT_ROOT / "data" / "processed" / "archetype_classifier.pkl"
SHAP_PATH = PROJECT_ROOT / "data" / "processed" / "shap_explanations.json"

FEATURE_COLUMNS = [
    "lighting_proxy",
    "activity_score",
    "connectivity_score",
    "main_road_proximity",
    "transit_proximity",
    "dead_end_penalty",
    "industrial_penalty",
]

TRAVEL_MODE_MAP = {"walking": 0, "cycling": 1, "cab": 2}
DESTINATION_TYPE_MAP = {"residential": 0, "commercial": 1, "transit": 2, "unknown": 3}


@dataclass
class RouteSummary:
    path: list[int]
    edges: list[dict[str, Any]]
    total_time: float
    mean_safety: float
    archetype: str | None = None


class NightSafeRouter:
    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        classifier_path: Path = CLASSIFIER_PATH,
        shap_path: Path = SHAP_PATH,
    ) -> None:
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing scored graph: {graph_path}")
        if not classifier_path.exists():
            raise FileNotFoundError(f"Missing archetype classifier: {classifier_path}")
        if not shap_path.exists():
            raise FileNotFoundError(f"Missing SHAP explanations: {shap_path}")

        self.graph = ox.load_graphml(graph_path)
        self.classifier = joblib.load(classifier_path)
        self.shap_explanations: dict[str, list[list[Any]]] = json.loads(shap_path.read_text())
        graph_crs = CRS.from_user_input(self.graph.graph.get("crs", "EPSG:4326"))
        self.coord_transformer = (
            None
            if graph_crs.to_epsg() == 4326
            else Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
        )
        self.inverse_transformer = (
            None
            if graph_crs.to_epsg() == 4326
            else Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)
        )
        self.service_area_bounds = self._compute_service_area_bounds()
        self.max_travel_time = max(
            (float(data.get("travel_time", 1.0)) for _, _, _, data in self.graph.edges(keys=True, data=True)),
            default=1.0,
        )
        self.edge_lookup = {
            str(data.get("edge_id", f"{u}_{v}_{key}")): data
            for u, v, key, data in self.graph.edges(keys=True, data=True)
        }
        self.agents = {archetype_id: self._load_agent(archetype_id) for archetype_id in ARCHETYPE_NAMES}
        self.envs = {
            archetype_id: NightRouteEnv(graph_path=graph_path, archetype_id=archetype_id)
            for archetype_id in ARCHETYPE_NAMES
        }
        logger.info(
            f"NightSafeRouter ready: nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, shap_entries={len(self.shap_explanations)}"
        )

    def _compute_service_area_bounds(self) -> dict[str, float]:
        xs = [float(node_data["x"]) for _, node_data in self.graph.nodes(data=True)]
        ys = [float(node_data["y"]) for _, node_data in self.graph.nodes(data=True)]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_lon, min_lat = (
            (min_x, min_y)
            if self.inverse_transformer is None
            else self.inverse_transformer.transform(min_x, min_y)
        )
        max_lon, max_lat = (
            (max_x, max_y)
            if self.inverse_transformer is None
            else self.inverse_transformer.transform(max_x, max_y)
        )
        return {
            "min_lat": float(min_lat),
            "min_lon": float(min_lon),
            "max_lat": float(max_lat),
            "max_lon": float(max_lon),
        }

    def _load_agent(self, archetype_id: int) -> PPOPolicyRouteAgent | QLearningRouteAgent:
        agent_path = agent_path_for_archetype(archetype_id)
        if not agent_path.exists():
            raise FileNotFoundError(f"Missing trained agent: {agent_path}")

        with zipfile.ZipFile(agent_path, "r") as archive:
            names = set(archive.namelist())
        if "agent.json" in names:
            return QLearningRouteAgent.load(agent_path, self.graph)
        env = NightRouteEnv(graph_path=GRAPH_PATH, archetype_id=archetype_id)
        return PPOPolicyRouteAgent.load(agent_path, archetype_id=archetype_id, env=env)

    def _snap_to_node(self, coords: list[float] | tuple[float, float]) -> int:
        lat, lon = float(coords[0]), float(coords[1])
        x_coord, y_coord = (
            (lon, lat)
            if self.coord_transformer is None
            else self.coord_transformer.transform(lon, lat)
        )
        return int(ox.distance.nearest_nodes(self.graph, X=x_coord, Y=y_coord))

    def _validate_coords_within_service_area(self, coords: list[float] | tuple[float, float], label: str) -> None:
        lat, lon = float(coords[0]), float(coords[1])
        bounds = self.service_area_bounds
        if bounds["min_lat"] <= lat <= bounds["max_lat"] and bounds["min_lon"] <= lon <= bounds["max_lon"]:
            return
        raise ValueError(
            f"{label} is outside the supported Bangalore service area "
            f"({bounds['min_lat']:.4f}-{bounds['max_lat']:.4f} lat, {bounds['min_lon']:.4f}-{bounds['max_lon']:.4f} lon)."
        )

    def _normalize_user_context(self, user_context: dict[str, Any]) -> dict[str, int]:
        travel_mode = user_context.get("travel_mode", 0)
        destination_type = user_context.get("destination_type", 3)
        hour_of_day = int(user_context.get("hour_of_day", 21))
        return {
            "travel_mode": TRAVEL_MODE_MAP.get(str(travel_mode), travel_mode) if not isinstance(travel_mode, int) else int(travel_mode),
            "hour_of_day": hour_of_day,
            "is_female": int(bool(user_context.get("is_female", 0))),
            "destination_type": DESTINATION_TYPE_MAP.get(str(destination_type), destination_type)
            if not isinstance(destination_type, int)
            else int(destination_type),
            "query_day_type": int(user_context.get("query_day_type", 0)),
        }

    def _edge_geometry(self, u: int, v: int, data: dict[str, Any]) -> LineString:
        geom = data.get("geometry")
        if isinstance(geom, LineString) and not geom.is_empty:
            return geom
        ux = float(self.graph.nodes[u]["x"])
        uy = float(self.graph.nodes[u]["y"])
        vx = float(self.graph.nodes[v]["x"])
        vy = float(self.graph.nodes[v]["y"])
        return LineString([(ux, uy), (vx, vy)])

    def _geometry_to_geojson(self, geometry: LineString) -> dict[str, Any]:
        if self.inverse_transformer is None:
            return mapping(geometry)
        geographic_geometry = transform(self.inverse_transformer.transform, geometry)
        return mapping(geographic_geometry)

    def _fallback_top_features(self, edge_data: dict[str, Any]) -> list[list[Any]]:
        signed_scores = {
            "lighting_proxy": float(edge_data.get("lighting_proxy", 0.0)) - 0.5,
            "activity_score": float(edge_data.get("activity_score", 0.0)) - 0.5,
            "connectivity_score": float(edge_data.get("connectivity_score", 0.0)) - 0.5,
            "main_road_proximity": float(edge_data.get("main_road_proximity", 0.0)) - 0.5,
            "transit_proximity": float(edge_data.get("transit_proximity", 0.0)) - 0.5,
            "dead_end_penalty": -float(edge_data.get("dead_end_penalty", 0.0)),
            "industrial_penalty": -float(edge_data.get("industrial_penalty", 0.0)),
        }
        ranked = sorted(signed_scores.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        return [[feature, float(value)] for feature, value in ranked]

    def _explain_edge(self, edge_data: dict[str, Any], top_features: list[list[Any]]) -> str:
        score = float(edge_data.get("safety_score", 50.0))
        negative_reasons: list[str] = []
        positive_reasons: list[str] = []
        for feature_name, shap_value in top_features:
            if feature_name == "dead_end_penalty" and float(edge_data.get("dead_end_penalty", 0.0)) > 0:
                negative_reasons.append("isolated dead-end segment")
            elif feature_name == "industrial_penalty" and float(edge_data.get("industrial_penalty", 0.0)) > 0:
                negative_reasons.append("industrial/service road with low activity")
            elif feature_name == "activity_score":
                (positive_reasons if shap_value >= 0 else negative_reasons).append(
                    "active nearby POIs" if shap_value >= 0 else "few nearby POIs"
                )
            elif feature_name == "lighting_proxy":
                (positive_reasons if shap_value >= 0 else negative_reasons).append(
                    "well-lit main road" if shap_value >= 0 else "weak lighting proxy"
                )
            elif feature_name == "main_road_proximity":
                (positive_reasons if shap_value >= 0 else negative_reasons).append(
                    "close to major roads" if shap_value >= 0 else "far from major roads"
                )
            elif feature_name == "transit_proximity":
                (positive_reasons if shap_value >= 0 else negative_reasons).append(
                    "near transit stops" if shap_value >= 0 else "far from transit"
                )
            elif feature_name == "connectivity_score":
                (positive_reasons if shap_value >= 0 else negative_reasons).append(
                    "well-connected junctions" if shap_value >= 0 else "low street connectivity"
                )

        if score < 45:
            reasons = ", ".join(dict.fromkeys(negative_reasons or positive_reasons or ["low inferred safety"]))
            return f"Low score: {reasons}"
        if score > 70:
            reasons = ", ".join(dict.fromkeys(positive_reasons or negative_reasons or ["strong inferred safety"]))
            return f"High score: {reasons}"
        reasons = ", ".join(dict.fromkeys((positive_reasons + negative_reasons) or ["mixed safety indicators"]))
        return f"Moderate score: {reasons}"

    def _segment_explanation(self, edge_data: dict[str, Any]) -> dict[str, Any]:
        edge_id = str(edge_data.get("edge_id"))
        top_features = self.shap_explanations.get(edge_id, self._fallback_top_features(edge_data))
        return {
            "score": float(edge_data.get("safety_score", edge_data.get("predicted_safety_score", 50.0))),
            "top_features": top_features,
            "explanation": self._explain_edge(edge_data, top_features),
            "evening_score": float(edge_data.get("evening_score", edge_data.get("safety_score", 50.0))),
            "night_score": float(edge_data.get("night_score", edge_data.get("safety_score", 50.0))),
        }

    def explain_segment(self, edge_id: str) -> dict[str, Any]:
        edge_data = self.edge_lookup.get(str(edge_id))
        if edge_data is None:
            raise KeyError(f"Unknown edge_id: {edge_id}")
        return self._segment_explanation(edge_data)

    def _edge_cost(self, edge_data: dict[str, Any], alpha: float) -> float:
        if "travel_time" not in edge_data and edge_data and all(isinstance(value, dict) for value in edge_data.values()):
            return min(self._edge_cost(candidate, alpha) for candidate in edge_data.values())
        norm_time = float(edge_data.get("travel_time", self.max_travel_time)) / self.max_travel_time
        norm_risk = 1.0 - float(edge_data.get("safety_score", 50.0)) / 100.0
        return float(alpha * norm_time + (1.0 - alpha) * norm_risk)

    def _node_path_to_route(
        self,
        node_path: list[int],
        route_type: str,
        archetype: str | None = None,
        alpha: float = 0.5,
    ) -> RouteSummary:
        if len(node_path) < 2:
            return RouteSummary(path=node_path, edges=[], total_time=0.0, mean_safety=0.0, archetype=archetype)

        route_edges: list[dict[str, Any]] = []
        for u, v in zip(node_path[:-1], node_path[1:], strict=True):
            edge_candidates = self.graph.get_edge_data(u, v)
            if not edge_candidates:
                continue
            best_key, best_edge = min(
                edge_candidates.items(),
                key=lambda item: float(item[1].get("travel_time", self.max_travel_time))
                if route_type == "fastest"
                else self._edge_cost(item[1], alpha),
            )
            edge_id = str(best_edge.get("edge_id", f"{u}_{v}_{best_key}"))
            geometry = self._edge_geometry(u, v, best_edge)
            explanation = self._segment_explanation(best_edge)
            route_edges.append(
                {
                    "edge_id": edge_id,
                    "u": int(u),
                    "v": int(v),
                    "travel_time": float(best_edge.get("travel_time", 0.0)),
                    "safety_score": float(best_edge.get("safety_score", 50.0)),
                    "geometry": self._geometry_to_geojson(geometry),
                    "top_features": explanation["top_features"],
                    "explanation": explanation["explanation"],
                }
            )

        total_time = float(sum(edge["travel_time"] for edge in route_edges))
        mean_safety = float(np.mean([edge["safety_score"] for edge in route_edges])) if route_edges else 0.0
        return RouteSummary(
            path=[int(node_id) for node_id in node_path],
            edges=route_edges,
            total_time=total_time,
            mean_safety=mean_safety,
            archetype=archetype,
        )

    def _shortest_path(self, origin_node: int, destination_node: int, alpha: float, route_type: str) -> RouteSummary:
        if route_type == "fastest":
            node_path = nx.shortest_path(self.graph, origin_node, destination_node, weight="travel_time")
        else:
            def _weight(_: int, __: int, edge_data: dict[str, Any]) -> float:
                return self._edge_cost(edge_data, alpha)

            node_path = nx.shortest_path(self.graph, origin_node, destination_node, weight=_weight)
        return self._node_path_to_route(node_path, route_type=route_type, alpha=alpha)

    def _run_agent_route(
        self,
        archetype_id: int,
        origin_node: int,
        destination_node: int,
        hour_of_day: int,
    ) -> RouteSummary:
        agent = self.agents[archetype_id]
        if isinstance(agent, QLearningRouteAgent):
            agent_result = agent.rollout(origin_node, destination_node, max_steps=300)
            node_path = [int(node) for node in agent_result.path]
        else:
            env = self.envs[archetype_id]
            agent_result = agent.rollout(
                env=env,
                origin_node=origin_node,
                destination_node=destination_node,
                time_of_day=float(hour_of_day),
            )
            node_path = [int(node) for node in agent_result.path]
        return self._node_path_to_route(
            node_path,
            route_type="balanced",
            archetype=ARCHETYPE_NAMES[archetype_id],
            alpha=0.5,
        )

    def _node_to_latlon(self, node_id: int) -> list[float]:
        x_coord = float(self.graph.nodes[node_id]["x"])
        y_coord = float(self.graph.nodes[node_id]["y"])
        lon, lat = (
            (x_coord, y_coord)
            if self.inverse_transformer is None
            else self.inverse_transformer.transform(x_coord, y_coord)
        )
        return [float(lat), float(lon)]

    def _to_payload(self, summary: RouteSummary) -> dict[str, Any]:
        payload = {
            "path": summary.path,
            "edges": summary.edges,
            "total_time": summary.total_time,
            "mean_safety": summary.mean_safety,
        }
        if summary.archetype is not None:
            payload["archetype"] = summary.archetype
        return payload

    def route(self, origin_coords: list[float], destination_coords: list[float], user_context: dict[str, Any]) -> dict[str, Any]:
        self._validate_coords_within_service_area(origin_coords, "Origin")
        self._validate_coords_within_service_area(destination_coords, "Destination")
        origin_node = self._snap_to_node(origin_coords)
        destination_node = self._snap_to_node(destination_coords)
        normalized_context = self._normalize_user_context(user_context)
        archetype_id = predict_archetype(normalized_context, classifier=self.classifier)

        fastest = self._shortest_path(origin_node, destination_node, alpha=1.0, route_type="fastest")
        balanced = self._shortest_path(origin_node, destination_node, alpha=0.5, route_type="balanced")
        safest = self._shortest_path(origin_node, destination_node, alpha=0.1, route_type="safest")
        agent_route = self._run_agent_route(
            archetype_id=archetype_id,
            origin_node=origin_node,
            destination_node=destination_node,
            hour_of_day=normalized_context["hour_of_day"],
        )

        pareto_frontier = []
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            summary = self._shortest_path(origin_node, destination_node, alpha=alpha, route_type="weighted")
            pareto_frontier.append(
                {
                    "alpha": alpha,
                    "eta_minutes": summary.total_time / 60.0,
                    "safety_score": summary.mean_safety,
                }
            )

        segment_explanations: dict[str, dict[str, Any]] = {}
        for route_summary in [fastest, balanced, safest, agent_route]:
            for edge in route_summary.edges:
                segment_explanations[edge["edge_id"]] = {
                    "score": edge["safety_score"],
                    "top_features": edge["top_features"],
                    "explanation": edge["explanation"],
                }

        return {
            "fastest": self._to_payload(fastest),
            "balanced": self._to_payload(balanced),
            "safest": self._to_payload(safest),
            "agent_route": self._to_payload(agent_route),
            "pareto_frontier": pareto_frontier,
            "segment_explanations": segment_explanations,
            "service_area_bounds": self.service_area_bounds,
        }


def main() -> None:
    router = NightSafeRouter()
    node_ids = list(router.graph.nodes())
    origin_node = node_ids[100]
    destination_node = node_ids[5000]
    for candidate in node_ids[5000:]:
        if nx.has_path(router.graph, origin_node, candidate):
            destination_node = candidate
            break
    origin = router._node_to_latlon(origin_node)
    destination = router._node_to_latlon(destination_node)
    result = router.route(
        origin_coords=origin,
        destination_coords=destination,
        user_context={
            "travel_mode": "walking",
            "hour_of_day": 22,
            "is_female": True,
            "destination_type": "residential",
            "query_day_type": 0,
        },
    )
    logger.info(
        "Route sanity check complete: "
        f"fastest_edges={len(result['fastest']['edges'])}, "
        f"balanced_edges={len(result['balanced']['edges'])}, "
        f"safest_edges={len(result['safest']['edges'])}, "
        f"agent_edges={len(result['agent_route']['edges'])}, "
        f"agent_archetype={result['agent_route']['archetype']}"
    )
    logger.info(f"Pareto points={result['pareto_frontier']}")


if __name__ == "__main__":
    main()
