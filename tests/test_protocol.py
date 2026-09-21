import pandas as pd

from src.eval import protocol


def _events(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["visitorid", "itemid"])


def test_build_relevant_items_groups_by_user() -> None:
    test_events = _events([(1, 10), (1, 20), (2, 30)])
    relevant = protocol.build_relevant_items(test_events)
    assert relevant == {1: {10, 20}, 2: {30}}


def test_eligible_users_only_includes_test_period_users() -> None:
    test_events = _events([(1, 10), (2, 20)])
    assert protocol.eligible_users(test_events) == {1, 2}


def test_seen_items_by_user_from_training() -> None:
    train_events = _events([(1, 10), (1, 11), (2, 20)])
    seen = protocol.seen_items_by_user(train_events)
    assert seen == {1: {10, 11}, 2: {20}}


def test_filter_seen_items_removes_training_items_by_default() -> None:
    recommended = [10, 11, 12]
    seen = {10, 11}
    filtered = protocol.filter_seen_items(recommended, seen)
    assert filtered == [12]


def test_filter_seen_items_can_be_disabled() -> None:
    recommended = [10, 11, 12]
    seen = {10, 11}
    filtered = protocol.filter_seen_items(recommended, seen, exclude_seen=False)
    assert filtered == [10, 11, 12]


def test_filter_seen_items_with_no_seen_items() -> None:
    recommended = [10, 11]
    filtered = protocol.filter_seen_items(recommended, set())
    assert filtered == [10, 11]
