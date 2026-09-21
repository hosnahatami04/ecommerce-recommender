from pathlib import Path

import pandas as pd
import pytest

from src.data import events


def _write_events_csv(path: Path, rows: list[tuple]) -> None:
    df = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"])
    df.to_csv(path, index=False)


def test_load_events_sorts_by_timestamp(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    base = events.KNOWN_MIN_TIMESTAMP_MS + 1000
    _write_events_csv(
        csv_path,
        [
            (base + 300, 1, "view", 10, None),
            (base + 100, 2, "view", 11, None),
            (base + 200, 1, "addtocart", 10, None),
        ],
    )
    df = events.load_events(csv_path)
    assert list(df["timestamp"]) == [base + 100, base + 200, base + 300]


def test_load_events_drops_exact_duplicates(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    base = events.KNOWN_MIN_TIMESTAMP_MS + 1000
    _write_events_csv(
        csv_path,
        [
            (base, 1, "view", 10, None),
            (base, 1, "view", 10, None),
            (base + 1, 2, "view", 11, None),
        ],
    )
    df = events.load_events(csv_path)
    assert len(df) == 2


def test_load_events_rejects_unknown_event_type(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    base = events.KNOWN_MIN_TIMESTAMP_MS + 1000
    _write_events_csv(csv_path, [(base, 1, "click", 10, None)])
    with pytest.raises(ValueError, match="Unexpected event type"):
        events.load_events(csv_path)


def test_load_events_rejects_out_of_range_timestamps(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    # Deliberately outside the known May-Sept 2015 range.
    _write_events_csv(csv_path, [(0, 1, "view", 10, None)])
    with pytest.raises(ValueError, match="out of expected"):
        events.load_events(csv_path)
