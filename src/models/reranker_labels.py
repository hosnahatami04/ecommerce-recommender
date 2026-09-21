"""Build leakage-safe labeled training data for the reranker.

Fits ALS + popularity on sub_train only, generates candidates for every
user active in the label window, and labels each candidate 1 if the
user actually interacted with it during the label window, else 0. Both
sub_train and label_window come from inside the original training
period -- validation and test are never referenced here.
"""

from __future__ import annotations

import pandas as pd

from src.eval.protocol import build_relevant_items, eligible_users
from src.models.als_model import ALSModel
from src.models.candidates import generate_candidates
from src.models.popularity import PopularityModel


def build_labeled_candidates(
    sub_train: pd.DataFrame,
    label_window: pd.DataFrame,
    pool_size: int = 200,
    als_model: ALSModel | None = None,
    popularity_model: PopularityModel | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns [visitorid, itemid, label].

    label_window's actual interactions define the positive labels.
    Models are fit on sub_train only unless already-fit instances are
    passed in (useful for tests / reuse).
    """
    if als_model is None:
        als_model = ALSModel().fit(sub_train)
    if popularity_model is None:
        popularity_model = PopularityModel().fit(sub_train)

    relevant_by_user = build_relevant_items(label_window)
    users = sorted(eligible_users(label_window))

    rows = []
    for user_id in users:
        candidates = generate_candidates(user_id, als_model, popularity_model, pool_size)
        relevant = relevant_by_user.get(user_id, set())
        for item_id in candidates:
            rows.append((user_id, item_id, 1 if item_id in relevant else 0))

    return pd.DataFrame(rows, columns=["visitorid", "itemid", "label"])
