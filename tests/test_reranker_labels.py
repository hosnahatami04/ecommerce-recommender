import pandas as pd

from src.models.als_model import ALSModel
from src.models.popularity import PopularityModel
from src.models.reranker_labels import build_labeled_candidates


def test_build_labeled_candidates_labels_actual_interactions_as_positive() -> None:
    sub_train = pd.DataFrame(
        {"visitorid": [1, 1], "itemid": [10, 20], "event": ["view", "view"]}
    )
    label_window = pd.DataFrame({"visitorid": [1], "itemid": [10]})

    als_model = ALSModel(factors=2, iterations=2).fit(sub_train)
    popularity_model = PopularityModel().fit(sub_train)

    labeled = build_labeled_candidates(
        sub_train,
        label_window,
        pool_size=5,
        als_model=als_model,
        popularity_model=popularity_model,
    )

    row = labeled[(labeled["visitorid"] == 1) & (labeled["itemid"] == 10)]
    assert row.iloc[0]["label"] == 1


def test_build_labeled_candidates_non_interacted_item_is_negative() -> None:
    sub_train = pd.DataFrame(
        {"visitorid": [1, 1], "itemid": [10, 20], "event": ["view", "view"]}
    )
    label_window = pd.DataFrame({"visitorid": [1], "itemid": [10]})

    als_model = ALSModel(factors=2, iterations=2).fit(sub_train)
    popularity_model = PopularityModel().fit(sub_train)

    labeled = build_labeled_candidates(
        sub_train,
        label_window,
        pool_size=5,
        als_model=als_model,
        popularity_model=popularity_model,
    )

    row = labeled[(labeled["visitorid"] == 1) & (labeled["itemid"] == 20)]
    if not row.empty:
        assert row.iloc[0]["label"] == 0


def test_build_labeled_candidates_only_includes_label_window_users() -> None:
    sub_train = pd.DataFrame(
        {"visitorid": [1, 2], "itemid": [10, 20], "event": ["view", "view"]}
    )
    # Only user 1 has an event in the label window.
    label_window = pd.DataFrame({"visitorid": [1], "itemid": [10]})

    als_model = ALSModel(factors=2, iterations=2).fit(sub_train)
    popularity_model = PopularityModel().fit(sub_train)

    labeled = build_labeled_candidates(
        sub_train,
        label_window,
        pool_size=5,
        als_model=als_model,
        popularity_model=popularity_model,
    )
    assert set(labeled["visitorid"]) == {1}
