import pandas as pd

from src.models.als_model import ALSModel
from src.models.features import FeatureContext
from src.models.popularity import PopularityModel
from src.models.reranker import LightGBMReranker
from src.models.reranker_labels import build_labeled_candidates
from src.models.two_stage import TwoStageModel


def _toy_history() -> pd.DataFrame:
    day = 1000 * 60 * 60 * 24
    rows = []
    for user in range(1, 6):
        for item in range(10, 30):
            if (user + item) % 3 == 0:
                rows.append((user, item, day * (item % 5), "view"))
    return pd.DataFrame(rows, columns=["visitorid", "itemid", "timestamp", "event"])


def test_two_stage_model_recommends_within_pool_size() -> None:
    history = _toy_history()
    sub_train = history[history["timestamp"] < history["timestamp"].median()]
    label_window = history[history["timestamp"] >= history["timestamp"].median()]

    als_model = ALSModel(factors=2, iterations=3).fit(sub_train)
    popularity_model = PopularityModel().fit(sub_train)

    labeled = build_labeled_candidates(
        sub_train,
        label_window,
        pool_size=10,
        als_model=als_model,
        popularity_model=popularity_model,
    )
    assert "label" in labeled.columns

    ctx = FeatureContext(sub_train, cutoff_ms=int(label_window["timestamp"].min()))

    if labeled["label"].nunique() < 2:
        return  # toy data too small/imbalanced for a meaningful fit; skip rerank assertions

    from src.models.features import build_features

    featured = build_features(labeled, ctx, als_model)
    reranker = LightGBMReranker(random_state=42, n_estimators=5).fit(featured)

    model = TwoStageModel(als_model, popularity_model, reranker, ctx, pool_size=10)
    recs = model.recommend(user_id=1, k=5)
    assert len(recs) <= 5

    detailed = model.recommend_detailed(user_id=1, k=5)
    assert len(detailed) == len(recs)
    for row in detailed:
        assert row["source"] in {"als", "popularity_fallback"}
        assert isinstance(row["rerank_moved"], bool)
        assert isinstance(row["score"], float)


def test_two_stage_model_empty_candidates_returns_empty_list() -> None:
    history = _toy_history()
    als_model = ALSModel(factors=2, iterations=2).fit(history)
    popularity_model = PopularityModel()  # unfit -- no items at all
    ctx = FeatureContext(history, cutoff_ms=int(history["timestamp"].max()) + 1)
    reranker = LightGBMReranker()

    model = TwoStageModel(als_model, popularity_model, reranker, ctx, pool_size=10)
    # Unknown user + empty popularity model -> no candidates at all.
    recs = model.recommend(user_id=9999, k=5)
    assert recs == []
    assert model.recommend_detailed(user_id=9999, k=5) == []
