"""Build the sparse user x item interaction matrix for ALS.

Built strictly from training events — not a single validation or test
event may contribute a cell to this matrix, or the model would be
learning from data it's later "predicting."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.weighting import apply_weights


class InteractionMatrix:
    """Sparse user x item confidence-weighted matrix, with id <-> index mappings."""

    def __init__(self, matrix: csr_matrix, user_ids: np.ndarray, item_ids: np.ndarray) -> None:
        self.matrix = matrix
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        self.idx_to_item_id = {i: iid for iid, i in self.item_id_to_idx.items()}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def build_interaction_matrix(train_events: pd.DataFrame) -> InteractionMatrix:
    """Build a confidence-weighted sparse user x item matrix from training events only.

    Weights are summed when a user has multiple events on the same item
    (e.g. viewed twice and added to cart once) — repeated interaction is
    itself a stronger signal of interest.
    """
    weighted = apply_weights(train_events)

    user_ids = np.sort(weighted["visitorid"].unique())
    item_ids = np.sort(weighted["itemid"].unique())

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    rows = weighted["visitorid"].map(user_id_to_idx).to_numpy()
    cols = weighted["itemid"].map(item_id_to_idx).to_numpy()
    data = weighted["weight"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(item_ids))
    )
    matrix.sum_duplicates()

    return InteractionMatrix(matrix, user_ids, item_ids)
