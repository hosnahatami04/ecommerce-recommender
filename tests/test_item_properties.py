from pathlib import Path

import pandas as pd

from src.data import item_properties


def _write_properties_csv(path: Path, rows: list[tuple]) -> None:
    df = pd.DataFrame(rows, columns=["timestamp", "itemid", "property", "value"])
    df.to_csv(path, index=False)


def test_load_category_history_filters_to_categoryid_only(tmp_path: Path) -> None:
    p1 = tmp_path / "p1.csv"
    p2 = tmp_path / "p2.csv"
    _write_properties_csv(
        p1,
        [
            (100, 1, "categoryid", "50"),
            (100, 1, "available", "1"),
        ],
    )
    _write_properties_csv(p2, [(200, 2, "categoryid", "60")])

    history = item_properties.load_category_history(p1, p2)
    assert len(history) == 2
    assert set(history["itemid"]) == {1, 2}
    assert list(history["categoryid"]) == [50, 60]


def test_category_as_of_uses_most_recent_before_cutoff() -> None:
    history = pd.DataFrame(
        {
            "itemid": [1, 1, 1],
            "timestamp": [100, 200, 300],
            "categoryid": [10, 20, 30],
        }
    )
    # Cutoff before the 3rd assignment -> should see categoryid=20, not 30.
    result = item_properties.category_as_of(history, cutoff_ms=250)
    assert result[1] == 20


def test_category_as_of_excludes_items_with_no_prior_assignment() -> None:
    history = pd.DataFrame({"itemid": [1], "timestamp": [500], "categoryid": [10]})
    result = item_properties.category_as_of(history, cutoff_ms=100)
    assert 1 not in result


def test_category_as_of_empty_history_returns_empty_dict() -> None:
    history = pd.DataFrame({"itemid": [], "timestamp": [], "categoryid": []})
    result = item_properties.category_as_of(history, cutoff_ms=1000)
    assert result == {}


def test_category_as_of_never_uses_future_assignment() -> None:
    # This is the leakage test: an assignment recorded exactly at or
    # after the cutoff must never appear in the result.
    history = pd.DataFrame({"itemid": [1], "timestamp": [1000], "categoryid": [99]})
    result = item_properties.category_as_of(history, cutoff_ms=1000)
    assert 1 not in result
