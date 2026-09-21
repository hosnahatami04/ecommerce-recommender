import json

from src.models.als_model import ALSModel


def test_als_recommend_output_is_json_serializable() -> None:
    """Regression test: model.recommend() must return plain ints, not numpy.int64,
    since results are always written to results/*.json."""
    import pandas as pd

    train = pd.DataFrame({"visitorid": [1, 1], "itemid": [10, 20], "event": ["view", "view"]})
    model = ALSModel(factors=2, iterations=2).fit(train)
    recs = model.recommend(1, 5)

    json.dumps({"recs": recs})  # raises TypeError if any element is numpy.int64
