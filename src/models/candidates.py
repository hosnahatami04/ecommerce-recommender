"""Stage 1 of the two-stage architecture: candidate generation.

ALS retrieves up to `pool_size` candidates per user. Users ALS has
never seen (zero training interactions, or an ALS score list shorter
than pool_size) are padded with the popularity model's ranked list, so
every user ends up with a full candidate list regardless of whether
ALS knows them.
"""

from __future__ import annotations

from src.models.als_model import ALSModel
from src.models.popularity import PopularityModel


def generate_candidates(
    user_id: int,
    als_model: ALSModel,
    popularity_model: PopularityModel,
    pool_size: int = 200,
) -> list[int]:
    """Return up to `pool_size` candidate item ids for user_id.

    ALS candidates come first (in ALS's own ranked order), then the
    popularity fallback fills any remaining slots with items ALS didn't
    already suggest, preserving popularity order.
    """
    als_candidates = als_model.recommend(user_id, pool_size)

    if len(als_candidates) >= pool_size:
        return als_candidates[:pool_size]

    seen_in_als = set(als_candidates)
    needed = pool_size - len(als_candidates)

    fallback_pool = popularity_model.recommend(user_id, pool_size * 2)
    fallback_fill = [item for item in fallback_pool if item not in seen_in_als][:needed]

    return als_candidates + fallback_fill
