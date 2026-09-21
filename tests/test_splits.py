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


def test_reranker_label_start_precedes_val_start() -> None:
    config = splits.load_split_config()
    assert config["reranker_label_start_ms"] < config["val_start_ms"]


def test_split_train_for_reranker_partitions_train_only() -> None:
    train = pd.DataFrame(
        {
            "timestamp": [10, 20, 50, 60, 90],
            "visitorid": [1] * 5,
            "event": ["view"] * 5,
            "itemid": list(range(5)),
        }
    )
    config = {"reranker_label_start_ms": 55}
    sub_train, label_window = splits.split_train_for_reranker(train, config=config)

    assert len(sub_train) + len(label_window) == len(train)
    assert sub_train["timestamp"].max() < 55
    assert label_window["timestamp"].min() >= 55


def test_split_train_for_reranker_never_sees_validation_or_test() -> None:
    # sub_train/label_window are built only from what's passed in as
    # train_events -- this test documents that validation/test rows
    # have no path into the function at all.
    train = pd.DataFrame(
        {"timestamp": [10, 20], "visitorid": [1, 1], "event": ["view", "view"], "itemid": [1, 2]}
    )
    validation = pd.DataFrame(
        {"timestamp": [30], "visitorid": [1], "event": ["view"], "itemid": [99]}
    )
    config = {"reranker_label_start_ms": 15}

    sub_train, label_window = splits.split_train_for_reranker(train, config=config)
    all_items = set(sub_train["itemid"]) | set(label_window["itemid"])
    assert 99 not in all_items
    del validation
