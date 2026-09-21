"""Load and clean the Retailrocket event log.

events.csv columns: timestamp (ms epoch), visitorid, event
(view / addtocart / transaction), itemid, transactionid (only set for
transaction events).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EVENTS_PATH = RAW_DIR / "events.csv"

EXPECTED_EVENT_TYPES = {"view", "addtocart", "transaction"}

# The dataset is documented as spanning roughly May-September 2015.
# Used as a sanity bound, not a hard filter.
KNOWN_MIN_TIMESTAMP_MS = pd.Timestamp("2015-05-01").value // 1_000_000
KNOWN_MAX_TIMESTAMP_MS = pd.Timestamp("2015-09-30").value // 1_000_000


def load_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    """Load the raw event log, drop exact duplicates, and sort by time.

    Returns a DataFrame sorted ascending by timestamp with columns:
    timestamp, visitorid, event, itemid, transactionid.
    """
    df = pd.read_csv(path)

    unknown_events = set(df["event"].unique()) - EXPECTED_EVENT_TYPES
    if unknown_events:
        raise ValueError(f"Unexpected event type(s) in data: {unknown_events}")

    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    if ts_min < KNOWN_MIN_TIMESTAMP_MS or ts_max > KNOWN_MAX_TIMESTAMP_MS:
        raise ValueError(
            f"Timestamps out of expected May-September 2015 range: "
            f"[{ts_min}, {ts_max}] vs expected "
            f"[{KNOWN_MIN_TIMESTAMP_MS}, {KNOWN_MAX_TIMESTAMP_MS}]"
        )

    df = df.drop_duplicates()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df
