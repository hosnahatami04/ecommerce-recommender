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
    return [item for item, _source in generate_candidates_with_source(
        user_id, als_model, popularity_model, pool_size
    )]


def generate_candidates_with_source(
    user_id: int,
    als_model: ALSModel,
    popularity_model: PopularityModel,
    pool_size: int = 200,
) -> list[tuple[int, str]]:
    """Same as generate_candidates, but pairs each item with its source.

    Source is "als" for items ALS retrieved, or "popularity_fallback"
    for items added to fill out the pool. Used by the API to report
    which stage produced each recommendation.
    """
    als_candidates = als_model.recommend(user_id, pool_size)

    if len(als_candidates) >= pool_size:
        return [(item, "als") for item in als_candidates[:pool_size]]

    seen_in_als = set(als_candidates)
    needed = pool_size - len(als_candidates)

    fallback_pool = popularity_model.recommend(user_id, pool_size * 2)
    fallback_fill = [item for item in fallback_pool if item not in seen_in_als][:needed]

    return [(item, "als") for item in als_candidates] + [
        (item, "popularity_fallback") for item in fallback_fill
    ]
