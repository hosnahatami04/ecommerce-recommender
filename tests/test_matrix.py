import pandas as pd

from src.models.matrix import build_interaction_matrix


def test_matrix_shape_matches_unique_users_and_items() -> None:
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 2],
            "itemid": [10, 20, 10],
            "event": ["view", "addtocart", "view"],
        }
    )
    im = build_interaction_matrix(train)
    assert im.matrix.shape == (2, 2)
    assert im.n_users == 2
    assert im.n_items == 2


def test_matrix_applies_confidence_weights() -> None:
    train = pd.DataFrame(
        {"visitorid": [1], "itemid": [10], "event": ["transaction"]},
    )
    im = build_interaction_matrix(train)
    user_idx = im.user_id_to_idx[1]
    item_idx = im.item_id_to_idx[10]
    assert im.matrix[user_idx, item_idx] == 5  # transaction weight


def test_matrix_sums_weights_for_repeated_interactions() -> None:
    # User 1 viewed item 10 twice and added it to cart once:
    # weight = 1 + 1 + 3 = 5
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 1],
            "itemid": [10, 10, 10],
            "event": ["view", "view", "addtocart"],
        }
    )
    im = build_interaction_matrix(train)
    user_idx = im.user_id_to_idx[1]
    item_idx = im.item_id_to_idx[10]
    assert im.matrix[user_idx, item_idx] == 5


def test_matrix_never_includes_validation_or_test_rows() -> None:
    # Simulates the leakage trap: only rows explicitly passed as
    # `train` may appear in the matrix, regardless of what other data
    # exists elsewhere in the caller's scope.
    train = pd.DataFrame({"visitorid": [1], "itemid": [10], "event": ["view"]})
    validation = pd.DataFrame({"visitorid": [1], "itemid": [99], "event": ["view"]})
    test = pd.DataFrame({"visitorid": [1], "itemid": [77], "event": ["view"]})

    im = build_interaction_matrix(train)

    # Only item 10 (from train) should ever appear as a column.
    assert set(im.item_id_to_idx.keys()) == {10}
    assert 99 not in im.item_id_to_idx
    assert 77 not in im.item_id_to_idx
    # validation/test are unused here on purpose -- this test documents
    # that build_interaction_matrix has no way to see them at all.
    del validation, test


def test_matrix_id_index_mappings_are_consistent() -> None:
    train = pd.DataFrame(
        {"visitorid": [5, 3], "itemid": [200, 100], "event": ["view", "view"]}
    )
    im = build_interaction_matrix(train)
    for uid, idx in im.user_id_to_idx.items():
        assert im.user_ids[idx] == uid
    for iid, idx in im.item_id_to_idx.items():
        assert im.idx_to_item_id[idx] == iid


def test_matrix_accepts_weights_override() -> None:
    train = pd.DataFrame({"visitorid": [1], "itemid": [10], "event": ["view"]})
    im = build_interaction_matrix(train, weights={"view": 99})
    user_idx = im.user_id_to_idx[1]
    item_idx = im.item_id_to_idx[10]
    assert im.matrix[user_idx, item_idx] == 99
