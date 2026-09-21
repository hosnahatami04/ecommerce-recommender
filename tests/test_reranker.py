import pandas as pd

from src.models.reranker import LightGBMReranker


def _toy_labeled_data() -> pd.DataFrame:
    # A simple separable signal: high als_score + high popularity -> label 1.
    rows = []
    for i in range(50):
        rows.append(
            {
                "visitorid": 1,
                "itemid": i,
                "als_score": 1.0,
                "item_popularity": 1.0,
                "item_popularity_trend": 1.0,
                "item_age_days": 5.0,
                "user_activity": 10,
                "category_affinity_match": 1,
                "days_since_last_event": 1.0,
                "label": 1,
            }
        )
        rows.append(
            {
                "visitorid": 1,
                "itemid": i + 1000,
                "als_score": 0.0,
                "item_popularity": 0.0,
                "item_popularity_trend": 0.0,
                "item_age_days": 100.0,
                "user_activity": 1,
                "category_affinity_match": 0,
                "days_since_last_event": 50.0,
                "label": 0,
            }
        )
    return pd.DataFrame(rows)


def test_reranker_fits_and_scores() -> None:
    data = _toy_labeled_data()
    model = LightGBMReranker(random_state=42, n_estimators=10).fit(data)
    ranked = model.rerank(data)
    assert "score" in ranked.columns
    assert len(ranked) == len(data)


def test_reranker_ranks_positive_signal_higher() -> None:
    data = _toy_labeled_data()
    model = LightGBMReranker(random_state=42, n_estimators=20).fit(data)
    ranked = model.rerank(data)

    top_10_items = set(ranked.head(10)["itemid"])
    # The strong-signal items (0-49) should dominate the top of the ranking.
    assert len(top_10_items & set(range(50))) >= 8


def test_reranker_sorted_descending_within_user() -> None:
    data = _toy_labeled_data()
    model = LightGBMReranker(random_state=42, n_estimators=10).fit(data)
    ranked = model.rerank(data)
    scores = ranked["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_reranker_rerank_before_fit_raises() -> None:
    model = LightGBMReranker()
    try:
        model.rerank(pd.DataFrame({"visitorid": [1], "itemid": [1]}))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_reranker_deterministic_with_fixed_seed() -> None:
    data = _toy_labeled_data()
    model_a = LightGBMReranker(random_state=42, n_estimators=10).fit(data)
    model_b = LightGBMReranker(random_state=42, n_estimators=10).fit(data)
    ranked_a = model_a.rerank(data)
    ranked_b = model_b.rerank(data)
    assert ranked_a["score"].tolist() == ranked_b["score"].tolist()
