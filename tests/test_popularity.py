import pandas as pd

from src.models.popularity import PopularityModel, TimeDecayedPopularityModel


def test_popularity_model_ranks_by_raw_count() -> None:
    train = pd.DataFrame({"itemid": [1, 1, 1, 2, 2, 3]})
    model = PopularityModel().fit(train)
    assert model.ranked_items[:3] == [1, 2, 3]


def test_popularity_model_recommends_same_list_regardless_of_user() -> None:
    train = pd.DataFrame({"itemid": [1, 1, 2]})
    model = PopularityModel().fit(train)
    assert model.recommend(user_id=1, k=2) == model.recommend(user_id=999, k=2)


def test_popularity_model_recommend_truncates_to_k() -> None:
    train = pd.DataFrame({"itemid": [1, 2, 3]})
    model = PopularityModel().fit(train)
    assert len(model.recommend(user_id=1, k=2)) == 2


def test_popularity_model_is_deterministic() -> None:
    train = pd.DataFrame({"itemid": [1, 1, 2, 2, 3]})
    model_a = PopularityModel().fit(train)
    model_b = PopularityModel().fit(train)
    assert model_a.recommend(1, 10) == model_b.recommend(1, 10)


def test_time_decayed_model_favors_recent_items() -> None:
    # Item 1: many old interactions. Item 2: fewer but very recent interactions.
    day_ms = 1000 * 60 * 60 * 24
    train = pd.DataFrame(
        {
            "itemid": [1] * 20 + [2] * 3,
            "timestamp": [0] * 20 + [100 * day_ms] * 3,
        }
    )
    model = TimeDecayedPopularityModel(half_life_days=1.0).fit(train)
    # Item 2's interactions are recent (age=0) so they decay far less than
    # item 1's interactions, which are 100 days old relative to max_ts.
    assert model.ranked_items[0] == 2


def test_time_decayed_model_is_deterministic() -> None:
    day_ms = 1000 * 60 * 60 * 24
    train = pd.DataFrame({"itemid": [1, 2, 2], "timestamp": [0, day_ms, day_ms]})
    model_a = TimeDecayedPopularityModel().fit(train)
    model_b = TimeDecayedPopularityModel().fit(train)
    assert model_a.recommend(1, 10) == model_b.recommend(1, 10)
