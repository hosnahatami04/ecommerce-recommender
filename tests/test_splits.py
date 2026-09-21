import pandas as pd

from src.data import splits


def _toy_config() -> dict:
    return {"val_start_ms": 100, "test_start_ms": 200}


def test_split_boundaries_are_strictly_ordered() -> None:
    events = pd.DataFrame(
        {
            "timestamp": [50, 99, 100, 150, 199, 200, 250],
            "visitorid": [1, 1, 1, 1, 1, 1, 1],
            "event": ["view"] * 7,
            "itemid": list(range(7)),
        }
    )
    train, val, test = splits.split_events(events, config=_toy_config())

    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].min() < test["timestamp"].min()
    assert train["timestamp"].max() < 100
    assert val["timestamp"].max() < 200
    assert test["timestamp"].min() >= 200


def test_split_partitions_all_rows_exactly_once() -> None:
    events = pd.DataFrame(
        {
            "timestamp": [50, 99, 100, 150, 199, 200, 250],
            "visitorid": [1] * 7,
            "event": ["view"] * 7,
            "itemid": list(range(7)),
        }
    )
    train, val, test = splits.split_events(events, config=_toy_config())
    assert len(train) + len(val) + len(test) == len(events)


def test_load_split_config_reads_real_config() -> None:
    config = splits.load_split_config()
    assert config["val_start_ms"] < config["test_start_ms"]
