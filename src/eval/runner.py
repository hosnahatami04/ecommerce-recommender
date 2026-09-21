"""Run a model through the evaluation protocol and produce a results dict.

Every model (popularity baseline, ALS, two-stage) is run through this
same function so results are directly comparable.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

import pandas as pd

from src.eval.metrics import (
    catalog_coverage,
    compute_popularity_percentiles,
    ndcg_at_k,
    popularity_bias,
    recall_at_k,
)
from src.eval.protocol import (
    build_relevant_items,
    eligible_users,
    filter_seen_items,
    seen_items_by_user,
)


class Recommender(Protocol):
    def recommend(self, user_id: int, k: int) -> list[int]: ...


def evaluate_model(
    model: Recommender,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    catalog_size: int,
    k_values: tuple[int, ...] = (10, 20),
    exclude_seen: bool = True,
    candidate_pool_size: int = 200,
    user_subset: set[int] | None = None,
) -> dict:
    """Run `model` through the fixed evaluation protocol.

    For each eligible user (>=1 test-period event), generate a candidate
    list of `candidate_pool_size` recommendations, filter out seen items,
    then compute Recall@k and NDCG@k for each k in k_values. Coverage and
    popularity-bias are computed once over each user's max(k_values)
    truncated list.

    user_subset, if given, restricts evaluation to that subset of
    otherwise-eligible users (e.g. one cold-start segment) while
    popularity ranks are still computed from the full train_events, so
    "popular" means the same thing across every segment.
    """
    relevant_by_user = build_relevant_items(test_events)
    seen_by_user = seen_items_by_user(train_events)
    users = sorted(eligible_users(test_events))
    if user_subset is not None:
        users = [u for u in users if u in user_subset]

    item_counts = Counter(train_events["itemid"])
    popularity_rank = compute_popularity_percentiles(item_counts)

    max_k = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    all_top_lists: list[list[int]] = []

    for user_id in users:
        raw_recs = model.recommend(user_id, candidate_pool_size)
        seen = seen_by_user.get(user_id, set())
        filtered = filter_seen_items(raw_recs, seen, exclude_seen=exclude_seen)

        relevant = relevant_by_user.get(user_id, set())
        for k in k_values:
            recall_sums[k] += recall_at_k(filtered, relevant, k)
            ndcg_sums[k] += ndcg_at_k(filtered, relevant, k)

        all_top_lists.append(filtered[:max_k])

    n_users = len(users)
    results = {
        "n_users_evaluated": n_users,
        "recall": {f"@{k}": recall_sums[k] / n_users if n_users else 0.0 for k in k_values},
        "ndcg": {f"@{k}": ndcg_sums[k] / n_users if n_users else 0.0 for k in k_values},
        "coverage": {
            f"@{k}": catalog_coverage(all_top_lists, catalog_size, k) for k in k_values
        },
        "popularity_bias": {
            f"@{k}": popularity_bias(all_top_lists, popularity_rank, k) for k in k_values
        },
    }
    return results
