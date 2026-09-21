"""Layer 3 -- the full two-stage model, behind the same recommend(user_id, k) interface.

Stage 1 (ALS + popularity fallback) retrieves candidates; stage 2
(LightGBM) reranks them using features stage 1 can't see. Composed
here so it can be run through the same evaluation protocol as the
popularity baseline and plain ALS.
"""

from __future__ import annotations

import pandas as pd

from src.models.als_model import ALSModel
from src.models.candidates import generate_candidates
from src.models.features import FeatureContext, build_features
from src.models.popularity import PopularityModel
from src.models.reranker import LightGBMReranker


class TwoStageModel:
    def __init__(
        self,
        als_model: ALSModel,
        popularity_model: PopularityModel,
        reranker: LightGBMReranker,
        feature_context: FeatureContext,
        pool_size: int = 200,
    ) -> None:
        self.als_model = als_model
        self.popularity_model = popularity_model
        self.reranker = reranker
        self.feature_context = feature_context
        self.pool_size = pool_size

    def recommend(self, user_id: int, k: int) -> list[int]:
        candidates = generate_candidates(
            user_id, self.als_model, self.popularity_model, self.pool_size
        )
        if not candidates:
            return []

        pairs = pd.DataFrame({"visitorid": [user_id] * len(candidates), "itemid": candidates})
        featured = build_features(pairs, self.feature_context, self.als_model)
        ranked = self.reranker.rerank(featured)

        return ranked["itemid"].tolist()[:k]
