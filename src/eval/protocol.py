"""The evaluation protocol every model in this project follows.

Pinned once, here, so every model (popularity baseline, ALS, two-stage)
is judged by the exact same rules:

- Which users are evaluated: only users with >= 1 event in the test period.
  A user with zero test-period activity has nothing to measure recall/NDCG
  against, so including them would just dilute every metric with
  undefined (0/0) cases.
- What counts as a hit: the recommended item appears in that user's set of
  test-period item ids (any event type — view, addtocart, or transaction).
- Seen-item exclusion: items a user already interacted with during
  training are excluded from their recommendations by default.
  Recommending something the user already bought/viewed pads every
  metric dishonestly — it's not a real recommendation.
"""

from __future__ import annotations

import pandas as pd


def build_relevant_items(test_events: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user to the set of item ids they interacted with in the test period."""
    return test_events.groupby("visitorid")["itemid"].apply(set).to_dict()


def eligible_users(test_events: pd.DataFrame) -> set[int]:
    """Users with at least one test-period event — the only users that get scored."""
    return set(test_events["visitorid"].unique())


def seen_items_by_user(train_events: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user to the set of item ids they interacted with during training.

    Used to exclude already-seen items from that user's recommendations.
    """
    return train_events.groupby("visitorid")["itemid"].apply(set).to_dict()


def filter_seen_items(
    recommended: list[int], seen: set[int], exclude_seen: bool = True
) -> list[int]:
    """Remove items the user already interacted with in training, if exclude_seen is True."""
    if not exclude_seen:
        return recommended
    return [item for item in recommended if item not in seen]
