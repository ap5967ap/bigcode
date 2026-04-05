from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classifier.archetype_classifier import (
    ADAPTIVE,
    ARCHETYPE_NAMES,
    COMFORT_SEEKER,
    EFFICIENCY_FIRST,
    VULNERABLE_SOLO,
    build_synthetic_dataset,
)
from routing.router import FEATURE_COLUMNS, NightSafeRouter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGE_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "edge_features.csv"
SCORED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "scored_graph.graphml"
MODEL_PATHS = {
    "default": PROJECT_ROOT / "data" / "processed" / "model_default.pkl",
    "evening": PROJECT_ROOT / "data" / "processed" / "model_evening.pkl",
    "night": PROJECT_ROOT / "data" / "processed" / "model_night.pkl",
}
CLASSIFIER_PATH = PROJECT_ROOT / "data" / "processed" / "archetype_classifier.pkl"
REPORT_PATH = PROJECT_ROOT / "data" / "evaluation_report.json"
RANDOM_SEED = 42


def _sample_route_pairs(router: NightSafeRouter, n_pairs: int = 50) -> list[tuple[list[float], list[float]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    undirected = router.graph.to_undirected()
    component_nodes = list(max(nx.connected_components(undirected), key=len))
    pairs: list[tuple[list[float], list[float]]] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < 5000:
        attempts += 1
        origin_node = int(rng.choice(component_nodes))
        destination_node = int(rng.choice(component_nodes))
        if origin_node == destination_node:
            continue
        try:
            path = nx.shortest_path(router.graph, origin_node, destination_node, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 5:
            continue
        pairs.append((router._node_to_latlon(origin_node), router._node_to_latlon(destination_node)))
    if len(pairs) < n_pairs:
        raise RuntimeError(f"Only sampled {len(pairs)} valid OD pairs")
    return pairs


def evaluate_safety_models(df: pd.DataFrame) -> dict[str, Any]:
    _, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    output: dict[str, Any] = {}
    target_columns = {"default": "safety_score", "evening": "evening_score", "night": "night_score"}
    for model_name, target_column in target_columns.items():
        model = joblib.load(MODEL_PATHS[model_name])
        preds = model.predict(test_df[FEATURE_COLUMNS])
        importances = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        output[model_name] = {
            "mae": float(mean_absolute_error(test_df[target_column], preds)),
            "rmse": float(np.sqrt(mean_squared_error(test_df[target_column], preds))),
            "feature_importance_ranking": [
                {"feature": feature_name, "importance": float(score)}
                for feature_name, score in importances
            ],
        }
    return output


def evaluate_classifier() -> dict[str, Any]:
    x, y = build_synthetic_dataset()
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    model = joblib.load(CLASSIFIER_PATH)
    preds = model.predict(x_test)
    return classification_report(
        y_test,
        preds,
        labels=[VULNERABLE_SOLO, COMFORT_SEEKER, EFFICIENCY_FIRST, ADAPTIVE],
        target_names=[ARCHETYPE_NAMES[i] for i in [VULNERABLE_SOLO, COMFORT_SEEKER, EFFICIENCY_FIRST, ADAPTIVE]],
        output_dict=True,
        zero_division=0,
    )


def evaluate_route_comparison(
    router: NightSafeRouter,
    route_pairs: list[tuple[list[float], list[float]]],
) -> dict[str, Any]:
    route_names = ["fastest", "balanced", "safest", "agent_route"]
    safety_values = {route_name: [] for route_name in route_names}
    eta_values = {route_name: [] for route_name in route_names}
    user_context = {
        "travel_mode": "walking",
        "hour_of_day": 22,
        "is_female": True,
        "destination_type": "residential",
        "query_day_type": 0,
    }
    for origin, destination in tqdm(route_pairs, desc="Route comparison"):
        result = router.route(origin_coords=origin, destination_coords=destination, user_context=user_context)
        for route_name in route_names:
            safety_values[route_name].append(float(result[route_name]["mean_safety"]))
            eta_values[route_name].append(float(result[route_name]["total_time"]))

    summary = {
        "mean_safety_score": {
            route_name: float(np.mean(values))
            for route_name, values in safety_values.items()
        },
        "mean_eta_seconds": {
            route_name: float(np.mean(values))
            for route_name, values in eta_values.items()
        },
    }
    fastest_safety = summary["mean_safety_score"]["fastest"]
    safest_safety = summary["mean_safety_score"]["safest"]
    fastest_eta = summary["mean_eta_seconds"]["fastest"]
    safest_eta = summary["mean_eta_seconds"]["safest"]
    summary["safety_gain_pct"] = float((safest_safety - fastest_safety) / max(fastest_safety, 1e-9) * 100.0)
    summary["eta_penalty_pct"] = float((safest_eta - fastest_eta) / max(fastest_eta, 1e-9) * 100.0)
    summary["summary_text"] = (
        "Safest route improved mean safety by "
        f"{summary['safety_gain_pct']:.2f}% with only {summary['eta_penalty_pct']:.2f}% ETA increase"
    )
    return summary


def _apply_score_map(base_graph: nx.MultiDiGraph, edge_scores: dict[str, float]) -> nx.MultiDiGraph:
    graph = base_graph.copy()
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_id = str(data.get("edge_id", f"{u}_{v}_{key}"))
        if edge_id in edge_scores:
            data["safety_score"] = float(edge_scores[edge_id])
    return graph


def _mean_safest_route_safety(graph: nx.MultiDiGraph, pairs: list[tuple[list[float], list[float]]], router: NightSafeRouter) -> float:
    safety_scores: list[float] = []
    for origin, destination in pairs:
        origin_node = router._snap_to_node(origin)
        destination_node = router._snap_to_node(destination)

        def _weight(_: int, __: int, edge_data: dict[str, Any]) -> float:
            if "travel_time" not in edge_data and edge_data and all(isinstance(value, dict) for value in edge_data.values()):
                return min(_weight(0, 0, candidate) for candidate in edge_data.values())
            norm_time = float(edge_data.get("travel_time", router.max_travel_time)) / router.max_travel_time
            norm_risk = 1.0 - float(edge_data.get("safety_score", 50.0)) / 100.0
            return 0.1 * norm_time + 0.9 * norm_risk

        try:
            path = nx.shortest_path(graph, origin_node, destination_node, weight=_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        route_edge_scores: list[float] = []
        for u, v in zip(path[:-1], path[1:], strict=True):
            edge_candidates = graph.get_edge_data(u, v) or {}
            if not edge_candidates:
                continue
            best_edge = min(edge_candidates.values(), key=lambda edge: _weight(u, v, edge))
            route_edge_scores.append(float(best_edge.get("safety_score", 50.0)))
        if route_edge_scores:
            safety_scores.append(float(np.mean(route_edge_scores)))
    return float(np.mean(safety_scores)) if safety_scores else 0.0


def evaluate_ablation(
    df: pd.DataFrame,
    router: NightSafeRouter,
    route_pairs: list[tuple[list[float], list[float]]],
) -> dict[str, Any]:
    night_model = joblib.load(MODEL_PATHS["night"])
    default_model = joblib.load(MODEL_PATHS["default"])
    full_scores = {
        edge_id: float(score)
        for edge_id, score in zip(df["edge_id"].astype(str), night_model.predict(df[FEATURE_COLUMNS]), strict=True)
    }
    full_graph = _apply_score_map(router.graph, full_scores)
    full_mean_safety = _mean_safest_route_safety(full_graph, route_pairs, router)

    ablations: dict[str, float] = {}
    for feature_name in ["activity_score", "connectivity_score"]:
        ablated_df = df.copy()
        ablated_df[feature_name] = 0.0
        scores = {
            edge_id: float(score)
            for edge_id, score in zip(
                ablated_df["edge_id"].astype(str),
                night_model.predict(ablated_df[FEATURE_COLUMNS]),
                strict=True,
            )
        }
        ablated_graph = _apply_score_map(router.graph, scores)
        ablated_mean = _mean_safest_route_safety(ablated_graph, route_pairs, router)
        ablations[f"without_{feature_name}"] = float(ablated_mean - full_mean_safety)

    static_scores = {
        edge_id: float(score)
        for edge_id, score in zip(df["edge_id"].astype(str), default_model.predict(df[FEATURE_COLUMNS]), strict=True)
    }
    static_graph = _apply_score_map(router.graph, static_scores)
    static_mean = _mean_safest_route_safety(static_graph, route_pairs, router)
    ablations["without_time_aware_scoring"] = float(static_mean - full_mean_safety)
    ablations["full_model_mean_safety"] = float(full_mean_safety)
    return ablations


def evaluate_latency(router: NightSafeRouter, route_pairs: list[tuple[list[float], list[float]]]) -> dict[str, float]:
    latencies_ms: list[float] = []
    user_context = {
        "travel_mode": "walking",
        "hour_of_day": 22,
        "is_female": True,
        "destination_type": "residential",
        "query_day_type": 0,
    }
    for origin, destination in tqdm(route_pairs, desc="Latency benchmark"):
        started = time.perf_counter()
        router.route(origin_coords=origin, destination_coords=destination, user_context=user_context)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def run_evaluation() -> dict[str, Any]:
    df = pd.read_csv(EDGE_FEATURES_PATH)
    router = NightSafeRouter()
    route_pairs = _sample_route_pairs(router, n_pairs=50)
    report = {
        "safety_score_model": evaluate_safety_models(df),
        "route_comparison": evaluate_route_comparison(router, route_pairs),
        "archetype_classifier": evaluate_classifier(),
        "ablation_study": evaluate_ablation(df, router, route_pairs),
        "route_computation_latency": evaluate_latency(router, route_pairs),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run_evaluation()
    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
