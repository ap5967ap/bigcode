from classifier.archetype_classifier import (
    ARCHETYPE_NAMES,
    build_synthetic_dataset,
    get_archetype_weights,
    predict_archetype,
    train_archetype_classifier,
)


def test_classifier_predicts_valid_archetype_and_weights() -> None:
    x, y = build_synthetic_dataset(200)
    assert x.shape == (200, 5)
    assert y.shape == (200,)

    model = train_archetype_classifier()
    archetype_id = predict_archetype(
        {
            "travel_mode": 0,
            "hour_of_day": 22,
            "is_female": 1,
            "destination_type": 0,
            "query_day_type": 0,
        },
        classifier=model,
    )
    assert archetype_id in ARCHETYPE_NAMES
    weights = get_archetype_weights(archetype_id, hour_of_day=22)
    assert set(weights) == {"alpha", "beta", "vulnerability_weight"}
