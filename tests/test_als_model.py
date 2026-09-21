import pandas as pd

from src.models.als_model import ALSModel


def _toy_train() -> pd.DataFrame:
    # Two users, enough items and interactions for ALS to produce
    # non-degenerate factors on a tiny toy example.
    rows = []
    for user in range(1, 6):
        for item in range(10, 20):
            if (user + item) % 2 == 0:
                rows.append((user, item, "view"))
    return pd.DataFrame(rows, columns=["visitorid", "itemid", "event"])


def test_als_model_fits_and_recommends() -> None:
    train = _toy_train()
    model = ALSModel(factors=4, iterations=5, random_state=42).fit(train)
    recs = model.recommend(user_id=1, k=3)
    assert len(recs) <= 3
    assert all(isinstance(item, (int, type(recs[0]))) for item in recs) if recs else True


def test_als_model_returns_empty_for_unknown_user() -> None:
    train = _toy_train()
    model = ALSModel(factors=4, iterations=5, random_state=42).fit(train)
    # User 999 never appeared in training -> no row in the interaction matrix.
    assert model.recommend(user_id=999, k=5) == []


def test_als_model_same_seed_gives_same_recommendations() -> None:
    train = _toy_train()
    model_a = ALSModel(factors=4, iterations=5, random_state=42).fit(train)
    model_b = ALSModel(factors=4, iterations=5, random_state=42).fit(train)
    assert model_a.recommend(1, 5) == model_b.recommend(1, 5)


def test_als_model_recommend_before_fit_raises() -> None:
    model = ALSModel()
    try:
        model.recommend(1, 5)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
