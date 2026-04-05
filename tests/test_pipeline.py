import pandas as pd

from pipeline.synthetic_labels import EDGE_FEATURES_PATH, FEATURE_COLUMNS


def test_edge_features_csv_has_expected_columns_and_scores() -> None:
    df = pd.read_csv(EDGE_FEATURES_PATH)
    expected_columns = [
        "edge_id",
        "u",
        "v",
        *FEATURE_COLUMNS,
        "safety_score",
        "evening_score",
        "night_score",
    ]
    assert df.columns.tolist() == expected_columns
    assert len(df) > 0
    assert df["safety_score"].between(0, 100).all()
    assert df["evening_score"].between(0, 100).all()
    assert df["night_score"].between(0, 100).all()
