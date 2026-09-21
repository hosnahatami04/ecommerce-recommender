import pandas as pd
import pytest

from src.data import weighting


def test_load_weights_matches_config() -> None:
    weights = weighting.load_weights()
    assert weights == {"view": 1, "addtocart": 3, "transaction": 5}


def test_apply_weights_maps_each_event_type() -> None:
    events = pd.DataFrame(
        {
            "event": ["view", "addtocart", "transaction"],
            "itemid": [1, 2, 3],
        }
    )
    out = weighting.apply_weights(events, weights={"view": 1, "addtocart": 3, "transaction": 5})
    assert list(out["weight"]) == [1, 3, 5]


def test_apply_weights_raises_on_unmapped_event_type() -> None:
    events = pd.DataFrame({"event": ["click"], "itemid": [1]})
    with pytest.raises(ValueError, match="No weight defined"):
        weighting.apply_weights(events, weights={"view": 1})
