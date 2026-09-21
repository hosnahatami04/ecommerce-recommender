"""Map raw event types to implicit-feedback confidence weights.

The weights themselves are a modeling assumption, not a fact — they live
in config/weights.json so they can be revisited and the whole evaluation
re-run without touching any model code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "weights.json"


def load_weights(path: Path = CONFIG_PATH) -> dict[str, int]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def apply_weights(events: pd.DataFrame, weights: dict[str, int] | None = None) -> pd.DataFrame:
    """Add a 'weight' column mapping each event's type to its confidence weight.

    Raises if any event type in the data has no entry in the weight mapping.
    """
    if weights is None:
        weights = load_weights()

    unmapped = set(events["event"].unique()) - set(weights.keys())
    if unmapped:
        raise ValueError(f"No weight defined for event type(s): {unmapped}")

    out = events.copy()
    out["weight"] = out["event"].map(weights)
    return out
