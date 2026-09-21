"""Per (user, candidate item) feature builder for the LightGBM reranker.

Every feature is computed only from data available strictly before
`cutoff_ms` -- the start of whatever window the candidates are being
labeled or scored for. This is what lets the reranker see signals ALS
can't (recency, trend, category affinity) without leaking the future.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.models.als_model import ALSModel


class FeatureContext:
    """Precomputed lookups shared across every (user, item) pair in one window.

    Building these once per window (rather than per-row) keeps feature
    generation fast even over hundreds of thousands of candidates.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        cutoff_ms: int,
        category_as_of: dict[int, int] | None = None,
    ) -> None:
        """history: all events strictly before cutoff_ms (e.g. sub_train)."""
        self.cutoff_ms = cutoff_ms

        self.item_counts = Counter(history["itemid"])
        self.max_item_count = max(self.item_counts.values()) if self.item_counts else 1

        recent_cutoff = cutoff_ms - 14 * 24 * 60 * 60 * 1000
        recent = history[history["timestamp"] >= recent_cutoff]
        self.recent_item_counts = Counter(recent["itemid"])

        self.item_first_seen_ms = history.groupby("itemid")["timestamp"].min().to_dict()

        self.user_activity_count = history.groupby("visitorid").size().to_dict()
        self.user_last_event_ms = history.groupby("visitorid")["timestamp"].max().to_dict()

        # User's category affinity: for each user, the category they
        # interact with most often, given the as-of-cutoff category map.
        self.category_as_of = category_as_of or {}
        if self.category_as_of:
            items_with_cat = history["itemid"].map(self.category_as_of)
            user_item_cat = pd.DataFrame(
                {"visitorid": history["visitorid"], "categoryid": items_with_cat}
            ).dropna()
            if not user_item_cat.empty:
                top_category = (
                    user_item_cat.groupby("visitorid")["categoryid"]
                    .agg(lambda s: s.value_counts().idxmax())
                    .to_dict()
                )
            else:
                top_category = {}
        else:
            top_category = {}
        self.user_top_category = top_category

    def item_popularity(self, item_id: int) -> float:
        return self.item_counts.get(item_id, 0) / self.max_item_count

    def item_popularity_trend(self, item_id: int) -> float:
        """Recent (last 14 days) interaction count vs. all-time count.

        > 1.0 means the item is trending up recently relative to its
        overall history; 0 means no recent interactions at all.
        """
        total = self.item_counts.get(item_id, 0)
        recent = self.recent_item_counts.get(item_id, 0)
        if total == 0:
            return 0.0
        return recent / total

    def item_age_days(self, item_id: int) -> float:
        first_seen = self.item_first_seen_ms.get(item_id)
        if first_seen is None:
            return 0.0
        return (self.cutoff_ms - first_seen) / (1000 * 60 * 60 * 24)

    def user_activity(self, user_id: int) -> int:
        return self.user_activity_count.get(user_id, 0)

    def days_since_last_event(self, user_id: int) -> float:
        last_event = self.user_last_event_ms.get(user_id)
        if last_event is None:
            return -1.0  # sentinel: user has no prior history at all
        return (self.cutoff_ms - last_event) / (1000 * 60 * 60 * 24)

    def category_affinity_match(self, user_id: int, item_id: int) -> int:
        """1 if item_id's category matches the user's most-interacted category."""
        user_category = self.user_top_category.get(user_id)
        item_category = self.category_as_of.get(item_id)
        if user_category is None or item_category is None:
            return 0
        return int(user_category == item_category)


FEATURE_NAMES = [
    "als_score",
    "item_popularity",
    "item_popularity_trend",
    "item_age_days",
    "user_activity",
    "category_affinity_match",
    "days_since_last_event",
]


def build_features(
    pairs: pd.DataFrame,
    context: FeatureContext,
    als_model: ALSModel,
) -> pd.DataFrame:
    """pairs: DataFrame with columns [visitorid, itemid] (and optionally 'label').

    Returns pairs with FEATURE_NAMES columns appended.
    """
    als_scores = _als_scores(pairs, als_model)

    out = pairs.copy()
    out["als_score"] = als_scores
    out["item_popularity"] = out["itemid"].map(context.item_popularity)
    out["item_popularity_trend"] = out["itemid"].map(context.item_popularity_trend)
    out["item_age_days"] = out["itemid"].map(context.item_age_days)
    out["user_activity"] = out["visitorid"].map(context.user_activity)
    out["days_since_last_event"] = out["visitorid"].map(context.days_since_last_event)
    out["category_affinity_match"] = [
        context.category_affinity_match(u, i)
        for u, i in zip(out["visitorid"], out["itemid"], strict=True)
    ]

    return out


def _als_scores(pairs: pd.DataFrame, als_model: ALSModel) -> list[float]:
    """Look up each pair's ALS affinity score, or 0.0 if either side is unknown to ALS."""
    if als_model.model is None or als_model.interaction_matrix is None:
        return [0.0] * len(pairs)

    im = als_model.interaction_matrix
    user_factors = als_model.model.user_factors
    item_factors = als_model.model.item_factors

    scores = []
    for user_id, item_id in zip(pairs["visitorid"], pairs["itemid"], strict=True):
        user_idx = im.user_id_to_idx.get(user_id)
        item_idx = im.item_id_to_idx.get(item_id)
        if user_idx is None or item_idx is None:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(user_factors[user_idx], item_factors[item_idx])))
    return scores
