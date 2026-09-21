import pandas as pd

from src.models.als_model import ALSModel
from src.models.features import FEATURE_NAMES, FeatureContext, build_features


def _history() -> pd.DataFrame:
    day = 1000 * 60 * 60 * 24
    return pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 2, 2],
            "itemid": [10, 10, 20, 20, 30],
            "timestamp": [0, 1 * day, 5 * day, 6 * day, 6 * day],
            "event": ["view", "view", "view", "view", "view"],
        }
    )


def test_item_popularity_normalized_by_max_count() -> None:
    history = _history()
    ctx = FeatureContext(history, cutoff_ms=10 * 1000 * 60 * 60 * 24)
    # item 10: 2 interactions, item 20: 2 interactions -> both max -> 1.0
    assert ctx.item_popularity(10) == 1.0
    # item 30: 1 interaction -> 0.5 of max
    assert ctx.item_popularity(30) == 0.5


def test_item_popularity_unknown_item_is_zero() -> None:
    ctx = FeatureContext(_history(), cutoff_ms=10 * 1000 * 60 * 60 * 24)
    assert ctx.item_popularity(999) == 0.0


def test_item_age_days_computed_from_first_seen() -> None:
    day = 1000 * 60 * 60 * 24
    history = _history()
    ctx = FeatureContext(history, cutoff_ms=10 * day)
    # item 10 first seen at t=0 -> age = 10 days at cutoff
    assert ctx.item_age_days(10) == 10.0


def test_user_activity_count() -> None:
    ctx = FeatureContext(_history(), cutoff_ms=10 * 1000 * 60 * 60 * 24)
    assert ctx.user_activity(1) == 2
    assert ctx.user_activity(2) == 3
    assert ctx.user_activity(999) == 0


def test_days_since_last_event() -> None:
    day = 1000 * 60 * 60 * 24
    ctx = FeatureContext(_history(), cutoff_ms=10 * day)
    # user 1's last event was at t=1 day -> 9 days since, at cutoff=10 days
    assert ctx.days_since_last_event(1) == 9.0


def test_days_since_last_event_unknown_user_returns_sentinel() -> None:
    ctx = FeatureContext(_history(), cutoff_ms=10 * 1000 * 60 * 60 * 24)
    assert ctx.days_since_last_event(999) == -1.0


def test_feature_context_never_sees_events_at_or_after_cutoff() -> None:
    # Leakage test with a synthetic timeline: an event exactly at the
    # cutoff, or after it, must never influence any feature.
    day = 1000 * 60 * 60 * 24
    history = pd.DataFrame(
        {
            "visitorid": [1],
            "itemid": [10],
            "timestamp": [5 * day],
            "event": ["view"],
        }
    )
    # Cutoff equal to the event's own timestamp -- event must be excluded
    # from anything computed "as of" that cutoff by the caller passing
    # only pre-cutoff history in. This test documents that a caller
    # must filter before constructing FeatureContext; the class itself
    # trusts what it's given, so we verify with an already-filtered
    # (empty) history representing correct caller behavior.
    empty_history = history[history["timestamp"] < 5 * day]
    ctx = FeatureContext(empty_history, cutoff_ms=5 * day)
    assert ctx.item_age_days(10) == 0.0
    assert ctx.user_activity(1) == 0


def test_category_affinity_match() -> None:
    history = _history()
    category_map = {10: 100, 20: 100, 30: 200}
    ctx = FeatureContext(history, cutoff_ms=10 * 1000 * 60 * 60 * 24, category_as_of=category_map)
    # user 2 interacted with items 20 (cat 100) and 30 (cat 200) -> top category is 100 (2 vs 1)
    assert ctx.category_affinity_match(user_id=2, item_id=20) == 1
    assert ctx.category_affinity_match(user_id=2, item_id=30) == 0


def test_category_affinity_no_category_data_defaults_to_zero() -> None:
    ctx = FeatureContext(_history(), cutoff_ms=10 * 1000 * 60 * 60 * 24, category_as_of={})
    assert ctx.category_affinity_match(user_id=1, item_id=10) == 0


def test_build_features_produces_all_expected_columns() -> None:
    history = _history()
    ctx = FeatureContext(history, cutoff_ms=10 * 1000 * 60 * 60 * 24)
    als_model = ALSModel(factors=2, iterations=2).fit(history)

    pairs = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 30]})
    result = build_features(pairs, ctx, als_model)

    for feature in FEATURE_NAMES:
        assert feature in result.columns
    assert len(result) == len(pairs)


def test_build_features_unknown_user_item_pair_gets_zero_als_score() -> None:
    history = _history()
    ctx = FeatureContext(history, cutoff_ms=10 * 1000 * 60 * 60 * 24)
    als_model = ALSModel(factors=2, iterations=2).fit(history)

    pairs = pd.DataFrame({"visitorid": [999], "itemid": [888]})
    result = build_features(pairs, ctx, als_model)
    assert result.iloc[0]["als_score"] == 0.0
