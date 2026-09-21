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


def split_train_for_reranker(
    train_events: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training window into (sub_train, label_window) for reranker labeling.

    The reranker needs its own supervised training data, and it must
    not be the validation or test set. sub_train is used to fit an ALS
    model; label_window is used to generate candidates for that ALS
    model and label them by whether the user actually interacted with
    them during label_window. Both pieces come entirely from the
    original train split -- validation and test are never touched.
    """
    if config is None:
        config = load_split_config()

    label_start = config["reranker_label_start_ms"]

    sub_train = train_events[train_events["timestamp"] < label_start]
    label_window = train_events[train_events["timestamp"] >= label_start]

    return sub_train.reset_index(drop=True), label_window.reset_index(drop=True)
