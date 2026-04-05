from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd
from loguru import logger
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "featured_graph.graphml"
EDGE_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "edge_features.csv"
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "lighting_proxy",
    "activity_score",
    "connectivity_score",
    "main_road_proximity",
    "transit_proximity",
    "dead_end_penalty",
    "industrial_penalty",
]


def _score_from_weights(edge: dict[str, Any], weights: dict[str, float], rng: np.random.Generator) -> float:
    raw_score = (
        weights["lighting_proxy"] * float(edge["lighting_proxy"])
        + weights["activity_score"] * float(edge["activity_score"])
        + weights["connectivity_score"] * float(edge["connectivity_score"])
        + weights["main_road_proximity"] * float(edge["main_road_proximity"])
        + weights["transit_proximity"] * float(edge["transit_proximity"])
        - weights["dead_end_penalty"] * float(edge["dead_end_penalty"])
        - weights["industrial_penalty"] * float(edge["industrial_penalty"])
    )
    noisy_score = np.clip(raw_score + rng.normal(0.0, 0.03), 0.0, 1.0)
    return float(noisy_score * 100.0)


def generate_synthetic_labels() -> pd.DataFrame:
    if not FEATURED_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing featured graph: {FEATURED_GRAPH_PATH}")

    graph = ox.load_graphml(FEATURED_GRAPH_PATH)
    rng = np.random.default_rng(RANDOM_SEED)

    default_weights = {
        "lighting_proxy": 0.30,
        "activity_score": 0.25,
        "connectivity_score": 0.20,
        "main_road_proximity": 0.15,
        "transit_proximity": 0.10,
        "dead_end_penalty": 0.15,
        "industrial_penalty": 0.10,
    }
    evening_weights = {
        **default_weights,
        "activity_score": default_weights["activity_score"] + 0.05,
        "transit_proximity": default_weights["transit_proximity"] + 0.03,
    }
    night_weights = {
        **default_weights,
        "activity_score": default_weights["activity_score"] - 0.10,
        "lighting_proxy": default_weights["lighting_proxy"] + 0.15,
    }

    records: list[dict[str, Any]] = []
    for u, v, key, data in tqdm(
        graph.edges(keys=True, data=True),
        total=graph.number_of_edges(),
        desc="Generating labels",
    ):
        edge_payload = {name: float(data.get(name, 0.0)) for name in FEATURE_COLUMNS}
        records.append(
            {
                "edge_id": data.get("edge_id", f"{u}_{v}_{key}"),
                "u": int(u),
                "v": int(v),
                **edge_payload,
                "safety_score": _score_from_weights(data, default_weights, rng),
                "evening_score": _score_from_weights(data, evening_weights, rng),
                "night_score": _score_from_weights(data, night_weights, rng),
            }
        )

    df = pd.DataFrame.from_records(records)
    EDGE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EDGE_FEATURES_PATH, index=False)
    logger.info(f"Saved {len(df)} edge labels to {EDGE_FEATURES_PATH}")
    logger.info(
        "Safety score summary: "
        f"default_mean={df['safety_score'].mean():.2f}, "
        f"evening_mean={df['evening_score'].mean():.2f}, "
        f"night_mean={df['night_score'].mean():.2f}"
    )
    return df


def main() -> None:
    generate_synthetic_labels()


if __name__ == "__main__":
    main()
