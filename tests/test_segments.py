import pandas as pd

from src.eval.segments import assign_segment, segment_users, user_training_counts


def test_assign_segment_boundaries() -> None:
    assert assign_segment(0) == "0"
    assert assign_segment(1) == "1-2"
    assert assign_segment(2) == "1-2"
    assert assign_segment(3) == "3-10"
    assert assign_segment(10) == "3-10"
    assert assign_segment(11) == "11+"
    assert assign_segment(500) == "11+"


def test_user_training_counts_includes_zero_for_unseen_users() -> None:
    train = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 20, 30]})
    counts = user_training_counts(train, all_test_users={1, 2, 3})
    assert counts == {1: 2, 2: 1, 3: 0}


def test_segment_users_groups_correctly() -> None:
    counts = {1: 0, 2: 1, 3: 2, 4: 3, 5: 10, 6: 11, 7: 500}
    segments = segment_users(counts)
    assert segments["0"] == {1}
    assert segments["1-2"] == {2, 3}
    assert segments["3-10"] == {4, 5}
    assert segments["11+"] == {6, 7}


def test_segment_users_covers_every_user_exactly_once() -> None:
    counts = {i: i for i in range(20)}
    segments = segment_users(counts)
    all_assigned = set()
    for users in segments.values():
        all_assigned |= users
    assert all_assigned == set(counts.keys())
