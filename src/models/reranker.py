"""Stage 2 of the two-stage architecture: LightGBM reranker.

Binary classification (interacted / didn't) on the candidates ALS
retrieves, sorted by predicted probability. Starting with the simplest
real reranker per the project's simplest-first agreement; LambdaRank is
a possible upgrade if this shows promise.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.models.features import FEATURE_NAMES


class LightGBMReranker:
    def __init__(self, random_state: int = 42, **lgb_params) -> None:
        self.random_state = random_state
        self.lgb_params = lgb_params
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, labeled_features: pd.DataFrame) -> "LightGBMReranker":
        """labeled_features: rows with FEATURE_NAMES columns plus a 'label' column."""
        x = labeled_features[FEATURE_NAMES]
        y = labeled_features["label"]

        params = {
            "objective": "binary",
            "random_state": self.random_state,
            "verbosity": -1,
            **self.lgb_params,
        }
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(x, y)
        return self

    def rerank(self, candidate_features: pd.DataFrame) -> pd.DataFrame:
        """candidate_features: rows with visitorid, itemid, and FEATURE_NAMES columns.

        Returns the same rows with a 'score' column (predicted
        interaction probability), sorted descending by score within
        each user.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before rerank().")

        x = candidate_features[FEATURE_NAMES]
        scores = self.model.predict_proba(x)[:, 1]

        out = candidate_features.copy()
        out["score"] = scores
        out = out.sort_values(["visitorid", "score"], ascending=[True, False])
        return out.reset_index(drop=True)
