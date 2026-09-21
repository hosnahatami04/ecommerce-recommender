"""Temporal train/validation/test split boundaries.

A random split is wrong for this problem: it lets the model see events
from the future relative to events it's tested on, which never happens
in production and inflates every metric. Instead we split strictly by
time — train on the past, validate and test on two disjoint 2-week
windows in the future, in that order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "split.json"


def load_split_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def split_events(
    events: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split events into (train, validation, test) by timestamp.

    train:      timestamp <  val_start_ms
    validation: val_start_ms <= timestamp < test_start_ms
    test:       timestamp >= test_start_ms
    """
    if config is None:
        config = load_split_config()

    val_start = config["val_start_ms"]
    test_start = config["test_start_ms"]

    train = events[events["timestamp"] < val_start]
    validation = events[(events["timestamp"] >= val_start) & (events["timestamp"] < test_start)]
    test = events[events["timestamp"] >= test_start]

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
