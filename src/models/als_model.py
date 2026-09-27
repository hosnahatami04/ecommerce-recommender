"""Layer 2 — collaborative filtering via implicit ALS.

Wraps implicit.als.AlternatingLeastSquares behind the same
model.recommend(user_id, k) interface the popularity baseline uses, so
both can be run through the same evaluation protocol.
"""

from __future__ import annotations

from implicit.als import AlternatingLeastSquares

from src.models.matrix import InteractionMatrix, build_interaction_matrix


class ALSModel:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.model: AlternatingLeastSquares | None = None
        self.interaction_matrix: InteractionMatrix | None = None

    def fit(self, train_events, weights: dict[str, int] | None = None) -> "ALSModel":
        self.interaction_matrix = build_interaction_matrix(train_events, weights=weights)
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        # implicit expects a user x item matrix of confidence weights.
        self.model.fit(self.interaction_matrix.matrix)
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Top-k item ids for user_id, or [] if the user is unknown to ALS.

        A user with zero training interactions has no row in the
        interaction matrix and ALS has nothing to base a score on for
        them — that's exactly the cold-start gap this model can't close
        (Phase 5's job to measure).
        """
        if self.model is None or self.interaction_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        user_idx = self.interaction_matrix.user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        item_indices, _scores = self.model.recommend(
            user_idx,
            self.interaction_matrix.matrix[user_idx],
            N=k,
            filter_already_liked_items=False,
        )
        return [
            int(self.interaction_matrix.idx_to_item_id[idx])
            for idx in item_indices
            if idx in self.interaction_matrix.idx_to_item_id
        ]
