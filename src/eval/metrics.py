"""Recommendation quality metrics, implemented by hand.

No metric library is used here — every formula is written out explicitly
so it can be explained line by line.
"""

from __future__ import annotations

import math
from collections import Counter


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-k recommendations.

    recall@k = |{recommended top-k} ∩ {relevant}| / |relevant|

    Returns 0.0 if the user has no relevant items (caller should exclude
    such users from the aggregate rather than let them silently zero it).
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    DCG@k = sum over positions i=1..k of (1 if item_i is relevant else 0) / log2(i + 1)
    IDCG@k = the best possible DCG@k, i.e. all relevant items packed into
             the first min(k, |relevant|) positions.
    NDCG@k = DCG@k / IDCG@k

    A hit at position 1 contributes 1/log2(2) = 1.0; a hit at position 9
    contributes 1/log2(10) ≈ 0.301 — earlier hits are worth strictly more.
    Returns 0.0 if the user has no relevant items.
    """
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def catalog_coverage(all_recommendations: list[list], catalog_size: int, k: int) -> float:
    """Fraction of the catalog that is ever recommended, in anyone's top-k.

    coverage = |union of all users' top-k recommended items| / |catalog|

    This is the diversity check: a model that always recommends the same
    20 bestsellers to everyone has near-zero coverage regardless of how
    accurate it looks on Recall/NDCG.
    """
    if catalog_size == 0:
        return 0.0
    recommended_union: set = set()
    for recs in all_recommendations:
        recommended_union.update(recs[:k])
    return len(recommended_union) / catalog_size


def popularity_bias(
    all_recommendations: list[list], item_popularity_rank: dict, k: int
) -> float:
    """Mean popularity percentile of recommended items (0 = obscure, 1 = most popular).

    item_popularity_rank maps item_id -> percentile in [0, 1], where 1.0 is
    the single most popular item in the training data and 0.0 is the least
    popular. This function averages that percentile over every
    recommended slot across every user's top-k list.

    A score near 1.0 means the model is mostly just handing back
    bestsellers; a lower score means it's surfacing a broader mix.
    Items with no known popularity rank (never seen in training) are
    skipped rather than treated as 0, so cold items don't silently drag
    the score down as an artifact of missing data.
    """
    scores = []
    for recs in all_recommendations:
        for item in recs[:k]:
            if item in item_popularity_rank:
                scores.append(item_popularity_rank[item])
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_popularity_percentiles(item_counts: Counter) -> dict:
    """Convert raw training-set interaction counts into percentile ranks in [0, 1].

    Ties (items with equal counts) receive the same percentile — computed
    as the fraction of items with a strictly lower count.
    """
    if not item_counts:
        return {}

    sorted_counts = sorted(item_counts.values())
    n = len(sorted_counts)

    def percentile_for(count: int) -> float:
        # Number of items with strictly fewer interactions than `count`,
        # divided by (n - 1) so the single most popular item lands at 1.0
        # and the single least popular lands at 0.0.
        import bisect

        rank = bisect.bisect_left(sorted_counts, count)
        if n == 1:
            return 1.0
        return rank / (n - 1)

    return {item: percentile_for(count) for item, count in item_counts.items()}
