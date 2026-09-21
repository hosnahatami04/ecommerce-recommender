"""Long-tail analysis: how recommendation frequency relates to item popularity rank.

Produces, for a given model's recommendations across all evaluated
users, how many times each item was recommended, alongside that item's
popularity rank in the training data. A model that only ever
recommends the most popular handful of items shows up here as a spike
concentrated at the lowest popularity ranks.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def recommendation_frequency(all_recommendations: list[list[int]], k: int) -> Counter:
    """Count how many times each item appears in anyone's top-k recommendations."""
    counts: Counter = Counter()
    for recs in all_recommendations:
        counts.update(recs[:k])
    return counts


def long_tail_table(
    all_recommendations: list[list[int]], train_events: pd.DataFrame, k: int
) -> pd.DataFrame:
    """Return a DataFrame with columns [itemid, popularity_rank, recommend_count].

    popularity_rank is 1 for the single most popular training item,
    2 for the next, and so on. recommend_count is how many times that
    item appeared in anyone's top-k recommendations (0 if never).
    """
    item_counts = train_events["itemid"].value_counts()
    popularity_rank = pd.Series(
        range(1, len(item_counts) + 1), index=item_counts.index, name="popularity_rank"
    )

    rec_counts = recommendation_frequency(all_recommendations, k)

    table = popularity_rank.reset_index()
    table.columns = ["itemid", "popularity_rank"]
    table["recommend_count"] = table["itemid"].map(rec_counts).fillna(0).astype(int)

    return table
