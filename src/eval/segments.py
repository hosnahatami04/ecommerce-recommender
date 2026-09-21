"""Segment users by training history length.

Averages hide failure: a model that does great for active users and
nothing for cold ones can still post a decent overall average. This
module defines the segments every KPI gets broken out by, so that
hiding place has nowhere left to go.
"""

from __future__ import annotations

import pandas as pd

SEGMENT_BOUNDARIES = [
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-10", 3, 10),
    ("11+", 11, None),
]


def user_training_counts(train_events: pd.DataFrame, all_test_users: set[int]) -> dict[int, int]:
    """Map every test-period user to their training event count (0 if unseen in training)."""
    counts = train_events.groupby("visitorid").size().to_dict()
    return {user_id: counts.get(user_id, 0) for user_id in all_test_users}


def assign_segment(training_count: int) -> str:
    for label, low, high in SEGMENT_BOUNDARIES:
        if high is None:
            if training_count >= low:
                return label
        elif low <= training_count <= high:
            return label
    raise ValueError(f"training_count {training_count} did not match any segment")


def segment_users(training_counts: dict[int, int]) -> dict[str, set[int]]:
    """Group user ids by segment label."""
    segments: dict[str, set[int]] = {label: set() for label, _, _ in SEGMENT_BOUNDARIES}
    for user_id, count in training_counts.items():
        segments[assign_segment(count)].add(user_id)
    return segments
