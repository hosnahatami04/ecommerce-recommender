import pandas as pd

from src.eval.runner import evaluate_model
from src.models.popularity import PopularityModel


def test_evaluate_model_excludes_seen_items_from_recommendations() -> None:
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 1],
            "itemid": [10, 10, 10],
            "timestamp": [1, 1, 1],
        }
    )
    test = pd.DataFrame({"visitorid": [1], "itemid": [20], "timestamp": [2]})

    model = PopularityModel().fit(train)
    # Model would recommend item 10 first (it's the only training item),
    # but user 1 already saw it in training, so it must be excluded.
    results = evaluate_model(model, train, test, catalog_size=2, k_values=(1,))
    assert results["n_users_evaluated"] == 1
    # Since the only candidate (10) gets filtered out, recall must be 0.
    assert results["recall"]["@1"] == 0.0


def test_evaluate_model_scores_hit_correctly() -> None:
    train = pd.DataFrame({"visitorid": [1], "itemid": [10], "timestamp": [1]})
    test = pd.DataFrame({"visitorid": [2], "itemid": [10], "timestamp": [2]})

    model = PopularityModel().fit(train)
    # User 2 has no training history, so item 10 isn't filtered as "seen".
    results = evaluate_model(model, train, test, catalog_size=1, k_values=(1,))
    assert results["recall"]["@1"] == 1.0
    assert results["ndcg"]["@1"] == 1.0


def test_evaluate_model_only_scores_eligible_users() -> None:
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "timestamp": [1, 1]})
    # Only user 1 has a test-period event; user 2 must not be scored.
    test = pd.DataFrame({"visitorid": [1], "itemid": [30], "timestamp": [2]})

    model = PopularityModel().fit(train)
    results = evaluate_model(model, train, test, catalog_size=3, k_values=(1,))
    assert results["n_users_evaluated"] == 1


def test_evaluate_model_respects_user_subset() -> None:
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "timestamp": [1, 1]})
    test = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "timestamp": [2, 2]})

    model = PopularityModel().fit(train)
    results = evaluate_model(
        model, train, test, catalog_size=2, k_values=(1,), user_subset={1}
    )
    assert results["n_users_evaluated"] == 1


def test_evaluate_model_user_subset_none_scores_everyone() -> None:
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "timestamp": [1, 1]})
    test = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20], "timestamp": [2, 2]})

    model = PopularityModel().fit(train)
    results = evaluate_model(model, train, test, catalog_size=2, k_values=(1,), user_subset=None)
    assert results["n_users_evaluated"] == 2
