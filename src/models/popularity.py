"""Layer 1 — Popularity baseline.

Recommend the most-interacted-with items in the training window to
every user, regardless of who they are. Every later model (ALS,
two-stage) is judged as a delta against this number.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


class PopularityModel:
    """Recommends the globally most popular items — same ranked list for everyone."""

    def __init__(self) -> None:
        self.ranked_items: list[int] = []
        self.item_counts: Counter = Counter()

    def fit(self, train_events: pd.DataFrame) -> "PopularityModel":
        self.item_counts = Counter(train_events["itemid"])
        self.ranked_items = [item for item, _ in self.item_counts.most_common()]
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Same top-k list for every user_id — popularity ignores identity."""
        return self.ranked_items[:k]


class TimeDecayedPopularityModel:
    """Popularity baseline where recent interactions count more than old ones.

    Each event's weight is decayed exponentially by its age (in days)
    relative to the most recent training timestamp:
        weight = 0.5 ** (age_days / half_life_days)
    An event exactly half_life_days old contributes half as much as a
    brand-new one. Items are then ranked by summed decayed weight
    instead of raw count.
    """

    def __init__(self, half_life_days: float = 14.0) -> None:
        self.half_life_days = half_life_days
        self.ranked_items: list[int] = []
        self.item_scores: dict[int, float] = {}

    def fit(self, train_events: pd.DataFrame) -> "TimeDecayedPopularityModel":
        max_ts = train_events["timestamp"].max()
        age_days = (max_ts - train_events["timestamp"]) / (1000 * 60 * 60 * 24)
        decayed_weight = 0.5 ** (age_days / self.half_life_days)

        scores = pd.Series(decayed_weight.values, index=train_events["itemid"].values)
        self.item_scores = scores.groupby(scores.index).sum().to_dict()
        self.ranked_items = sorted(self.item_scores, key=self.item_scores.get, reverse=True)
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        return self.ranked_items[:k]
