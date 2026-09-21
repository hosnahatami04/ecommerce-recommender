import math
from collections import Counter

import pytest

from src.eval import metrics

# --- recall_at_k ---


def test_recall_at_k_perfect_hit() -> None:
    # All 2 relevant items are in the top 2 recommended -> recall = 1.0
    assert metrics.recall_at_k([10, 20, 30], {10, 20}, k=2) == pytest.approx(1.0)


def test_recall_at_k_partial_hit() -> None:
    # 1 of 2 relevant items appears in top-2 -> recall = 0.5
    assert metrics.recall_at_k([10, 30, 20], {10, 20}, k=2) == pytest.approx(0.5)


def test_recall_at_k_no_hit() -> None:
    assert metrics.recall_at_k([30, 40], {10, 20}, k=2) == pytest.approx(0.0)


def test_recall_at_k_no_relevant_items_returns_zero() -> None:
    assert metrics.recall_at_k([10, 20], set(), k=2) == 0.0


def test_recall_at_k_truncates_to_k() -> None:
    # Relevant item 20 is at position 3, outside top-2 -> not counted.
    assert metrics.recall_at_k([10, 30, 20], {20}, k=2) == pytest.approx(0.0)


# --- ndcg_at_k ---
# Hand-computed on paper: 3 users, 5 items (A, B, C, D, E).


def test_ndcg_at_k_hit_at_first_position_scores_one() -> None:
    # relevant = {A}, recommended = [A, B, C]
    # DCG = 1/log2(2) = 1.0 ; IDCG (1 relevant, best case at position 1) = 1.0
    # NDCG = 1.0
    recommended = ["A", "B", "C"]
    relevant = {"A"}
    result = metrics.ndcg_at_k(recommended, relevant, k=3)
    assert result == pytest.approx(1.0)


def test_ndcg_at_k_hit_at_later_position_scores_less_than_one() -> None:
    # relevant = {C}, recommended = [A, B, C]
    # DCG = 1/log2(4) = 0.5  (position 3: i=3, log2(3+1)=log2(4)=2)
    # IDCG (1 relevant, best case at position 1) = 1/log2(2) = 1.0
    # NDCG = 0.5
    recommended = ["A", "B", "C"]
    relevant = {"C"}
    result = metrics.ndcg_at_k(recommended, relevant, k=3)
    expected_dcg = 1.0 / math.log2(4)
    expected_idcg = 1.0 / math.log2(2)
    assert result == pytest.approx(expected_dcg / expected_idcg)
    assert result == pytest.approx(0.5)


def test_ndcg_at_k_two_relevant_items_mixed_positions() -> None:
    # relevant = {A, D}, recommended = [B, A, C, D, E], k=5
    # Hits at position 2 (A) and position 4 (D).
    # DCG = 1/log2(3) + 1/log2(5)
    # IDCG = best case: both relevant items packed at positions 1,2
    #      = 1/log2(2) + 1/log2(3)
    recommended = ["B", "A", "C", "D", "E"]
    relevant = {"A", "D"}
    result = metrics.ndcg_at_k(recommended, relevant, k=5)
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert result == pytest.approx(dcg / idcg)


def test_ndcg_at_k_no_relevant_items_returns_zero() -> None:
    assert metrics.ndcg_at_k(["A", "B"], set(), k=2) == 0.0


def test_ndcg_at_k_no_hits_returns_zero() -> None:
    assert metrics.ndcg_at_k(["A", "B"], {"Z"}, k=2) == pytest.approx(0.0)


# --- catalog_coverage ---


def test_catalog_coverage_full_coverage() -> None:
    # 2 users, top-2 each, union covers all 4 catalog items.
    all_recs = [[1, 2], [3, 4]]
    assert metrics.catalog_coverage(all_recs, catalog_size=4, k=2) == pytest.approx(1.0)


def test_catalog_coverage_partial_coverage() -> None:
    # Both users get the same 2 items recommended -> only 2 of 10 items covered.
    all_recs = [[1, 2], [1, 2]]
    assert metrics.catalog_coverage(all_recs, catalog_size=10, k=2) == pytest.approx(0.2)


def test_catalog_coverage_respects_k_truncation() -> None:
    # Only the first k=1 items of each list count.
    all_recs = [[1, 2, 3], [4, 5, 6]]
    assert metrics.catalog_coverage(all_recs, catalog_size=10, k=1) == pytest.approx(0.2)


def test_catalog_coverage_empty_catalog_returns_zero() -> None:
    assert metrics.catalog_coverage([[1]], catalog_size=0, k=1) == 0.0


# --- popularity_bias ---


def test_popularity_bias_all_most_popular() -> None:
    ranks = {1: 1.0, 2: 0.5, 3: 0.0}
    all_recs = [[1, 1], [1, 1]]
    assert metrics.popularity_bias(all_recs, ranks, k=2) == pytest.approx(1.0)


def test_popularity_bias_mixed_scores() -> None:
    ranks = {1: 1.0, 2: 0.0}
    all_recs = [[1, 2]]
    assert metrics.popularity_bias(all_recs, ranks, k=2) == pytest.approx(0.5)


def test_popularity_bias_skips_unknown_items() -> None:
    ranks = {1: 1.0}
    all_recs = [[1, 999]]  # item 999 has no known popularity rank
    assert metrics.popularity_bias(all_recs, ranks, k=2) == pytest.approx(1.0)


def test_popularity_bias_no_scored_items_returns_zero() -> None:
    ranks: dict = {}
    all_recs = [[1, 2]]
    assert metrics.popularity_bias(all_recs, ranks, k=2) == 0.0


# --- compute_popularity_percentiles ---


def test_compute_popularity_percentiles_orders_correctly() -> None:
    # Item A: 10 interactions (least popular), B: 20, C: 30 (most popular).
    counts = Counter({"A": 10, "B": 20, "C": 30})
    percentiles = metrics.compute_popularity_percentiles(counts)
    assert percentiles["A"] == pytest.approx(0.0)
    assert percentiles["B"] == pytest.approx(0.5)
    assert percentiles["C"] == pytest.approx(1.0)


def test_compute_popularity_percentiles_ties_get_equal_rank() -> None:
    counts = Counter({"A": 10, "B": 10, "C": 30})
    percentiles = metrics.compute_popularity_percentiles(counts)
    assert percentiles["A"] == percentiles["B"]
    assert percentiles["C"] == pytest.approx(1.0)


def test_compute_popularity_percentiles_empty_returns_empty() -> None:
    assert metrics.compute_popularity_percentiles(Counter()) == {}


def test_compute_popularity_percentiles_single_item_is_top() -> None:
    counts = Counter({"A": 5})
    percentiles = metrics.compute_popularity_percentiles(counts)
    assert percentiles["A"] == pytest.approx(1.0)
