from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import osmnx as ox
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGE_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "edge_features.csv"
FEATURED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "featured_graph.graphml"
SCORED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "scored_graph.graphml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_default.pkl"
EVENING_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_evening.pkl"
NIGHT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_night.pkl"
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
TARGETS = {
    "default": ("safety_score", DEFAULT_MODEL_PATH),
    "evening": ("evening_score", EVENING_MODEL_PATH),
    "night": ("night_score", NIGHT_MODEL_PATH),
}
RANDOM_SEED = 42


def _objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    preds = model.predict(x_valid)
    return float(mean_absolute_error(y_valid, preds))


def _train_target_model(df: pd.DataFrame, target_column: str, model_path: Path) -> tuple[XGBRegressor, dict[str, Any]]:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    train_inner, valid_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED)

    x_train = train_inner[FEATURE_COLUMNS]
    y_train = train_inner[target_column]
    x_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[target_column]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_column]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda trial: _objective(trial, x_train, y_train, x_valid, y_valid), n_trials=50, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    model = XGBRegressor(**best_params)
    model.fit(train_df[FEATURE_COLUMNS], train_df[target_column], verbose=False)
    preds = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    joblib.dump(model, model_path)
    logger.info(f"Saved {target_column} model to {model_path}")
    logger.info(f"{target_column} test metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, best_params={study.best_params}")
    return model, {"mae": mae, "rmse": rmse, "test_df": test_df}


def _build_shap_explanations(model: XGBRegressor, test_df: pd.DataFrame) -> dict[str, list[list[Any]]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df[FEATURE_COLUMNS])
    explanations: dict[str, list[list[Any]]] = {}
    for edge_id, row_values in zip(test_df["edge_id"].astype(str).tolist(), shap_values, strict=True):
        top_idx = np.argsort(np.abs(row_values))[::-1][:3]
        explanations[edge_id] = [
            [FEATURE_COLUMNS[int(idx)], float(row_values[int(idx)])]
            for idx in top_idx
        ]
    SHAP_PATH.write_text(json.dumps(explanations, indent=2))
    logger.info(f"Saved SHAP explanations for {len(explanations)} test edges to {SHAP_PATH}")
    return explanations


def _attach_predictions_to_graph(model: XGBRegressor, graph_df: pd.DataFrame) -> None:
    if not FEATURED_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing featured graph: {FEATURED_GRAPH_PATH}")
    graph = ox.load_graphml(FEATURED_GRAPH_PATH)
    predictions = model.predict(graph_df[FEATURE_COLUMNS])
    by_edge_id = dict(zip(graph_df["edge_id"].astype(str), predictions, strict=True))

    for u, v, key, data in tqdm(
        graph.edges(keys=True, data=True),
        total=graph.number_of_edges(),
        desc="Attaching predicted scores",
    ):
        edge_id = str(data.get("edge_id", f"{u}_{v}_{key}"))
        data["predicted_safety_score"] = float(np.clip(by_edge_id[edge_id], 0.0, 100.0))
        data["safety_score"] = data["predicted_safety_score"]

    ox.save_graphml(graph, filepath=SCORED_GRAPH_PATH)
    logger.info(f"Saved scored graph to {SCORED_GRAPH_PATH}")


def train_safety_models() -> dict[str, dict[str, float]]:
    if not EDGE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing edge features file: {EDGE_FEATURES_PATH}")

    df = pd.read_csv(EDGE_FEATURES_PATH)
    metrics: dict[str, dict[str, float]] = {}
    default_model: XGBRegressor | None = None
    default_test_df: pd.DataFrame | None = None

    for label, (target_column, model_path) in TARGETS.items():
        logger.info(f"Training {label} XGBoost safety model for target={target_column}")
        model, result = _train_target_model(df, target_column, model_path)
        metrics[label] = {"mae": result["mae"], "rmse": result["rmse"]}
        if label == "default":
            default_model = model
            default_test_df = result["test_df"]

    if default_model is None or default_test_df is None:
        raise RuntimeError("Default model was not trained")
    _build_shap_explanations(default_model, default_test_df)
    _attach_predictions_to_graph(default_model, df)
    return metrics


def main() -> None:
    metrics = train_safety_models()
    logger.info(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
