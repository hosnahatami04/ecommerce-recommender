import pandas as pd

from src.eval.longtail import long_tail_table, recommendation_frequency


def test_recommendation_frequency_counts_across_users() -> None:
    all_recs = [[10, 20], [10, 30], [10, 20]]
    freq = recommendation_frequency(all_recs, k=2)
    assert freq[10] == 3
    assert freq[20] == 2
    assert freq[30] == 1


def test_recommendation_frequency_respects_k() -> None:
    all_recs = [[10, 20, 30]]
    freq = recommendation_frequency(all_recs, k=1)
    assert freq[10] == 1
    assert 20 not in freq
    assert 30 not in freq


def test_long_tail_table_ranks_by_training_popularity() -> None:
    train = pd.DataFrame({"itemid": [10, 10, 10, 20, 20, 30]})
    all_recs = [[10]]
    table = long_tail_table(all_recs, train, k=1)

    row_10 = table[table["itemid"] == 10].iloc[0]
    row_20 = table[table["itemid"] == 20].iloc[0]
    row_30 = table[table["itemid"] == 30].iloc[0]

    assert row_10["popularity_rank"] == 1
    assert row_20["popularity_rank"] == 2
    assert row_30["popularity_rank"] == 3
    assert row_10["recommend_count"] == 1
    assert row_20["recommend_count"] == 0
    assert row_30["recommend_count"] == 0


def test_long_tail_table_includes_every_training_item() -> None:
    train = pd.DataFrame({"itemid": [1, 2, 3, 4]})
    table = long_tail_table([], train, k=10)
    assert len(table) == 4
    assert (table["recommend_count"] == 0).all()
