"""Leakage-safe lookup of item category assignments over time.

item_properties logs each item's properties as a timestamped stream --
a "categoryid" property can change over time for the same item. Any
feature that uses an item's category must only see the categoryid that
was in effect *before* the cutoff it's being computed for, or it leaks
future information into the reranker's training data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PART1_PATH = RAW_DIR / "item_properties_part1.csv"
PART2_PATH = RAW_DIR / "item_properties_part2.csv"


def load_category_history(
    part1_path: Path = PART1_PATH, part2_path: Path = PART2_PATH
) -> pd.DataFrame:
    """Load every timestamped categoryid assignment across both property files.

    Returns columns [itemid, timestamp, categoryid], sorted by timestamp.
    """
    p1 = pd.read_csv(part1_path)
    p2 = pd.read_csv(part2_path)
    combined = pd.concat([p1, p2], ignore_index=True)

    cat_rows = combined[combined["property"] == "categoryid"].copy()
    cat_rows = cat_rows.rename(columns={"value": "categoryid"})
    cat_rows["categoryid"] = cat_rows["categoryid"].astype(int)
    cat_rows = cat_rows[["itemid", "timestamp", "categoryid"]].sort_values("timestamp")

    return cat_rows.reset_index(drop=True)


def category_as_of(category_history: pd.DataFrame, cutoff_ms: int) -> dict[int, int]:
    """Map item_id -> categoryid, using only assignments recorded before cutoff_ms.

    When an item has multiple assignments before the cutoff, the most
    recent one (by timestamp) wins. Items with no assignment before the
    cutoff are absent from the returned mapping.
    """
    visible = category_history[category_history["timestamp"] < cutoff_ms]
    if visible.empty:
        return {}
    latest = visible.sort_values("timestamp").groupby("itemid", as_index=True).tail(1)
    return dict(zip(latest["itemid"], latest["categoryid"], strict=True))
