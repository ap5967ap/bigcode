from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = PROJECT_ROOT / "data" / "processed" / "archetype_classifier.pkl"
RANDOM_SEED = 42

VULNERABLE_SOLO = 0
COMFORT_SEEKER = 1
EFFICIENCY_FIRST = 2
ADAPTIVE = 3

ARCHETYPE_NAMES = {
    VULNERABLE_SOLO: "Vulnerable Solo",
    COMFORT_SEEKER: "Comfort Seeker",
    EFFICIENCY_FIRST: "Efficiency First",
    ADAPTIVE: "Adaptive",
}


def _sample_features(n_samples: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    travel_mode = rng.integers(0, 3, size=n_samples)
    hour_of_day = rng.integers(0, 24, size=n_samples)
    is_female = rng.integers(0, 2, size=n_samples)
    destination_type = rng.integers(0, 4, size=n_samples)
    query_day_type = rng.integers(0, 2, size=n_samples)
    return np.column_stack([travel_mode, hour_of_day, is_female, destination_type, query_day_type]).astype(np.float32)


def _rule_label(row: np.ndarray, rng: np.random.Generator) -> int:
    travel_mode, hour_of_day, is_female, destination_type, query_day_type = row.astype(int).tolist()
    if travel_mode == 0 and is_female == 1 and hour_of_day >= 20:
        return VULNERABLE_SOLO if rng.random() < 0.9 else COMFORT_SEEKER
    if travel_mode == 2:
        return EFFICIENCY_FIRST if rng.random() < 0.8 else ADAPTIVE
    if 19 <= hour_of_day <= 22 and destination_type == 1:
        return COMFORT_SEEKER if rng.random() < 0.85 else ADAPTIVE
    if hour_of_day >= 22 or hour_of_day <= 4:
        return ADAPTIVE if rng.random() < 0.7 else VULNERABLE_SOLO
    if travel_mode == 1 and destination_type == 2:
        return ADAPTIVE if rng.random() < 0.65 else COMFORT_SEEKER
    if query_day_type == 1 and destination_type == 1:
        return COMFORT_SEEKER if rng.random() < 0.7 else ADAPTIVE
    if travel_mode == 0 and hour_of_day >= 18:
        return COMFORT_SEEKER if rng.random() < 0.75 else VULNERABLE_SOLO
    return COMFORT_SEEKER if rng.random() < 0.55 else EFFICIENCY_FIRST


def build_synthetic_dataset(n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    x = _sample_features(n_samples)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    y = np.array([_rule_label(row, rng) for row in x], dtype=np.int64)
    flip_mask = rng.random(n_samples) < 0.15
    random_labels = rng.integers(0, 4, size=int(flip_mask.sum()))
    y[flip_mask] = random_labels
    return x, y


def train_archetype_classifier() -> Pipeline:
    x, y = build_synthetic_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    report = classification_report(
        y_test,
        preds,
        labels=[VULNERABLE_SOLO, COMFORT_SEEKER, EFFICIENCY_FIRST, ADAPTIVE],
        target_names=[ARCHETYPE_NAMES[i] for i in [VULNERABLE_SOLO, COMFORT_SEEKER, EFFICIENCY_FIRST, ADAPTIVE]],
        digits=4,
        zero_division=0,
    )
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASSIFIER_PATH)
    logger.info(f"Saved archetype classifier to {CLASSIFIER_PATH}")
    logger.info(f"Classification report:\n{report}")
    return model


def encode_user_context(user_context: dict[str, Any]) -> np.ndarray:
    return np.array(
        [[
            int(user_context.get("travel_mode", 0)),
            int(user_context.get("hour_of_day", 21)),
            int(user_context.get("is_female", 0)),
            int(user_context.get("destination_type", 3)),
            int(user_context.get("query_day_type", 0)),
        ]],
        dtype=np.float32,
    )


def predict_archetype(user_context: dict[str, Any], classifier: Pipeline | None = None) -> int:
    model = classifier if classifier is not None else joblib.load(CLASSIFIER_PATH)
    return int(model.predict(encode_user_context(user_context))[0])


def get_archetype_weights(archetype_id: int, hour_of_day: int | None = None) -> dict[str, float]:
    if archetype_id == VULNERABLE_SOLO:
        return {"alpha": 0.2, "beta": 0.8, "vulnerability_weight": 1.5}
    if archetype_id == COMFORT_SEEKER:
        return {"alpha": 0.45, "beta": 0.55, "vulnerability_weight": 1.0}
    if archetype_id == EFFICIENCY_FIRST:
        return {"alpha": 0.8, "beta": 0.2, "vulnerability_weight": 0.3}
    if archetype_id == ADAPTIVE:
        hour = 21 if hour_of_day is None else int(hour_of_day)
        if 22 <= hour or hour <= 4:
            return {"alpha": 0.35, "beta": 0.65, "vulnerability_weight": 1.2}
        if 19 <= hour <= 21:
            return {"alpha": 0.45, "beta": 0.55, "vulnerability_weight": 1.0}
        return {"alpha": 0.65, "beta": 0.35, "vulnerability_weight": 0.6}
    raise ValueError(f"Unknown archetype_id: {archetype_id}")


def main() -> None:
    model = train_archetype_classifier()
    sample_context = {
        "travel_mode": 0,
        "hour_of_day": 22,
        "is_female": 1,
        "destination_type": 0,
        "query_day_type": 0,
    }
    archetype_id = predict_archetype(sample_context, classifier=model)
    logger.info(
        f"Sample context predicted archetype={ARCHETYPE_NAMES[archetype_id]} "
        f"weights={get_archetype_weights(archetype_id, sample_context['hour_of_day'])}"
    )


if __name__ == "__main__":
    main()
