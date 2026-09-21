import pandas as pd

from src.models.als_model import ALSModel
from src.models.candidates import generate_candidates, generate_candidates_with_source
from src.models.popularity import PopularityModel


def test_candidates_pads_with_popularity_for_unknown_als_user() -> None:
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "event": ["view", "view"]})
    als_model = ALSModel(factors=2, iterations=2).fit(train)
    popularity_model = PopularityModel().fit(train)

    # User 999 is unknown to ALS -> candidates must come entirely from popularity.
    candidates = generate_candidates(999, als_model, popularity_model, pool_size=2)
    assert len(candidates) == 2
    assert set(candidates) <= {10, 20}


def test_candidates_never_returns_more_than_pool_size() -> None:
    train = pd.DataFrame(
        {"visitorid": [1] * 3, "itemid": [10, 20, 30], "event": ["view"] * 3}
    )
    als_model = ALSModel(factors=2, iterations=2).fit(train)
    popularity_model = PopularityModel().fit(train)

    candidates = generate_candidates(1, als_model, popularity_model, pool_size=2)
    assert len(candidates) <= 2


def test_candidates_fallback_does_not_duplicate_als_items() -> None:
    train = pd.DataFrame(
        {"visitorid": [1] * 5, "itemid": [10, 20, 30, 40, 50], "event": ["view"] * 5}
    )
    als_model = ALSModel(factors=2, iterations=2).fit(train)
    popularity_model = PopularityModel().fit(train)

    candidates = generate_candidates(1, als_model, popularity_model, pool_size=5)
    assert len(candidates) == len(set(candidates))


def test_candidates_with_source_unknown_user_all_fallback() -> None:
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "event": ["view", "view"]})
    als_model = ALSModel(factors=2, iterations=2).fit(train)
    popularity_model = PopularityModel().fit(train)

    result = generate_candidates_with_source(999, als_model, popularity_model, pool_size=2)
    assert all(source == "popularity_fallback" for _item, source in result)


def test_candidates_with_source_matches_generate_candidates() -> None:
    train = pd.DataFrame(
        {"visitorid": [1] * 3, "itemid": [10, 20, 30], "event": ["view"] * 3}
    )
    als_model = ALSModel(factors=2, iterations=2).fit(train)
    popularity_model = PopularityModel().fit(train)

    plain = generate_candidates(1, als_model, popularity_model, pool_size=3)
    with_source = generate_candidates_with_source(1, als_model, popularity_model, pool_size=3)
    assert plain == [item for item, _source in with_source]
