from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import osmnx as ox
from loguru import logger
from shapely.geometry import LineString, Point, base
from shapely.strtree import STRtree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_GRAPH_PATH = PROJECT_ROOT / "data" / "raw" / "road_graph.graphml"
RAW_POI_PATH = PROJECT_ROOT / "data" / "raw" / "pois.geojson"
FEATURED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "featured_graph.graphml"

HIGH_LIGHTING = {"motorway", "trunk", "primary"}
MID_LIGHTING = {"secondary", "tertiary"}
LOW_LIGHTING = {"residential", "living_street"}
MIN_LIGHTING = {"service", "path", "track", "unclassified", "footway", "steps", "cycleway"}
MAIN_ROADS = {"primary", "secondary"}
COMMERCIAL_AMENITIES = {"restaurant", "cafe", "bank", "fuel", "pharmacy"}
TRANSIT_VALUES = {"stop_position", "platform"}


def _as_point(geom: base.BaseGeometry | None) -> Point | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Point):
        return geom
    return geom.representative_point()


def _edge_midpoint(graph: nx.MultiDiGraph, u: int, v: int, data: dict[str, Any]) -> Point:
    geom = data.get("geometry")
    if isinstance(geom, LineString) and not geom.is_empty:
        return geom.interpolate(0.5, normalized=True)
    ux = float(graph.nodes[u]["x"])
    uy = float(graph.nodes[u]["y"])
    vx = float(graph.nodes[v]["x"])
    vy = float(graph.nodes[v]["y"])
    return LineString([(ux, uy), (vx, vy)]).interpolate(0.5, normalized=True)


def _highway_type(highway: Any) -> str:
    if isinstance(highway, list) and highway:
        return str(highway[0])
    return str(highway) if highway is not None else "unclassified"


def _base_lighting_proxy(highway: str) -> float:
    if highway in HIGH_LIGHTING:
        return 1.0
    if highway in MID_LIGHTING:
        return 0.7
    if highway in LOW_LIGHTING:
        return 0.4
    if highway in MIN_LIGHTING:
        return 0.2
    return 0.4


def _build_point_tree(points: list[Point]) -> STRtree | None:
    return STRtree(points) if points else None


def _build_main_road_geoms(graph: nx.MultiDiGraph) -> list[LineString]:
    lines: list[LineString] = []
    for _, _, data in graph.edges(data=True):
        highway = _highway_type(data.get("highway"))
        if highway not in MAIN_ROADS:
            continue
        geom = data.get("geometry")
        if isinstance(geom, LineString) and not geom.is_empty:
            lines.append(geom)
    return lines


def _count_nearby(tree: STRtree | None, point: Point, radius_m: float) -> int:
    if tree is None:
        return 0
    return int(len(tree.query(point.buffer(radius_m))))


def _nearest_distance(tree: STRtree | None, geoms: list[base.BaseGeometry], point: Point) -> float:
    if tree is None or not geoms:
        return float("inf")
    nearest_idx = tree.nearest(point)
    if nearest_idx is None:
        return float("inf")
    return float(point.distance(geoms[int(nearest_idx)]))


def _load_projected_inputs() -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    if not RAW_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing graph file: {RAW_GRAPH_PATH}")
    if not RAW_POI_PATH.exists():
        raise FileNotFoundError(f"Missing POI file: {RAW_POI_PATH}")

    graph = ox.load_graphml(RAW_GRAPH_PATH)
    graph = ox.project_graph(graph)
    pois = gpd.read_file(RAW_POI_PATH)
    if pois.empty:
        pois = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    if pois.crs is None:
        pois.set_crs("EPSG:4326", inplace=True)
    pois = pois.to_crs(graph.graph["crs"])
    return graph, pois


def _commercial_mask(pois: gpd.GeoDataFrame) -> Iterable[bool]:
    for _, row in pois.iterrows():
        amenity = row.get("amenity")
        shop = row.get("shop")
        yield bool(shop) or str(amenity) in COMMERCIAL_AMENITIES


def _transit_mask(pois: gpd.GeoDataFrame) -> Iterable[bool]:
    for _, row in pois.iterrows():
        yield str(row.get("public_transport")) in TRANSIT_VALUES


def engineer_features() -> nx.MultiDiGraph:
    graph, pois = _load_projected_inputs()
    poi_points: list[Point] = []
    commercial_points: list[Point] = []
    transit_points: list[Point] = []
    commercial_flags = list(_commercial_mask(pois))
    transit_flags = list(_transit_mask(pois))
    for idx, geom in enumerate(pois.geometry):
        point = _as_point(geom)
        if point is None:
            continue
        poi_points.append(point)
        if commercial_flags[idx]:
            commercial_points.append(point)
        if transit_flags[idx]:
            transit_points.append(point)

    main_road_geoms = _build_main_road_geoms(graph)
    undirected_degree = dict(graph.to_undirected().degree())

    poi_tree = _build_point_tree(poi_points)
    commercial_tree = _build_point_tree(commercial_points)
    transit_tree = _build_point_tree(transit_points)
    main_road_tree = STRtree(main_road_geoms) if main_road_geoms else None

    logger.info(
        f"Engineering features for {graph.number_of_edges()} edges using "
        f"{len(poi_points)} POIs, {len(commercial_points)} commercial POIs, {len(transit_points)} transit POIs"
    )

    for u, v, key, data in tqdm(
        graph.edges(keys=True, data=True),
        total=graph.number_of_edges(),
        desc="Feature engineering",
    ):
        highway = _highway_type(data.get("highway"))
        midpoint = _edge_midpoint(graph, u, v, data)

        poi_count_150m = _count_nearby(poi_tree, midpoint, 150.0)
        commercial_bonus = 0.1 if _count_nearby(commercial_tree, midpoint, 100.0) >= 3 else 0.0
        lighting_proxy = min(1.0, _base_lighting_proxy(highway) + commercial_bonus)

        activity_score = min(1.0, math.log1p(poi_count_150m) / math.log(20.0))

        degree_u = graph.in_degree(u) + graph.out_degree(u)
        degree_v = graph.in_degree(v) + graph.out_degree(v)
        connectivity_score = min(1.0, ((degree_u + degree_v) / 2.0) / 8.0)

        dist_to_main = _nearest_distance(main_road_tree, main_road_geoms, midpoint)
        main_road_proximity = max(0.0, 1.0 - dist_to_main / 500.0)

        dist_to_transit = _nearest_distance(transit_tree, transit_points, midpoint)
        transit_proximity = max(0.0, 1.0 - dist_to_transit / 300.0)

        undirected_degree_u = undirected_degree.get(u, 0)
        undirected_degree_v = undirected_degree.get(v, 0)
        dead_end_penalty = 1.0 if undirected_degree_u <= 1 or undirected_degree_v <= 1 else 0.0

        industrial_penalty = 1.0 if highway == "service" and poi_count_150m == 0 else 0.0

        data["edge_id"] = data.get("edge_id", f"{u}_{v}_{key}")
        data["highway"] = highway
        data["lighting_proxy"] = float(lighting_proxy)
        data["activity_score"] = float(activity_score)
        data["connectivity_score"] = float(connectivity_score)
        data["main_road_proximity"] = float(main_road_proximity)
        data["transit_proximity"] = float(transit_proximity)
        data["dead_end_penalty"] = float(dead_end_penalty)
        data["industrial_penalty"] = float(industrial_penalty)
        data["poi_count_150m"] = int(poi_count_150m)

    FEATURED_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=FEATURED_GRAPH_PATH)
    logger.info(f"Saved featured graph to {FEATURED_GRAPH_PATH}")
    return graph


def main() -> None:
    engineer_features()


if __name__ == "__main__":
    main()
