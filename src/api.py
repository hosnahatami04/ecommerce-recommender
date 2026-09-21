"""FastAPI serving layer for the two-stage recommender.

GET /recommend/{user_id}?k=10 -- ranked recommendations, with each
item's stage (als-retrieved / popularity-fallback / rerank position
moved) so the response is demo-worthy, not a black box.

GET /health -- artifact load status.

Unknown users (no training history, or never seen by any model) get
the plain popularity list back with "fallback": true rather than a
404 -- this is what production systems do, and it keeps the endpoint
always useful instead of erroring on exactly the users who need
recommendations most.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.models.artifacts import load_artifacts  # noqa: E402
from src.models.two_stage import TwoStageModel  # noqa: E402

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        als_model, popularity_model, reranker, feature_context = load_artifacts()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to load model artifacts: {exc}. "
            "The API cannot start without trained artifacts."
        ) from exc

    _state["als_model"] = als_model
    _state["popularity_model"] = popularity_model
    _state["two_stage_model"] = TwoStageModel(
        als_model, popularity_model, reranker, feature_context
    )
    yield
    _state.clear()


app = FastAPI(title="Ecommerce Recommender API", lifespan=lifespan)


class RecommendedItem(BaseModel):
    item_id: int
    score: float
    source: str
    rerank_moved: bool


class RecommendResponse(BaseModel):
    user_id: int
    items: list[RecommendedItem]
    fallback: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "two_stage_model" in _state else "not_ready"}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = Query(default=10, ge=1, le=200)) -> RecommendResponse:
    two_stage_model: TwoStageModel = _state["two_stage_model"]
    als_model = _state["als_model"]
    popularity_model = _state["popularity_model"]

    known_to_als = user_id in als_model.interaction_matrix.user_id_to_idx
    known_to_popularity = popularity_model.ranked_items != []

    if not known_to_als and not known_to_popularity:
        raise HTTPException(
            status_code=503, detail="No models available to serve a recommendation"
        )

    # A user unknown to ALS is served entirely from the popularity
    # fallback inside the two-stage pipeline (see candidates.py) --
    # flagged here so the response is honest about where it came from,
    # rather than a 404 for exactly the users who need a
    # recommendation the most (see the README's unknown-user decision).
    detailed = two_stage_model.recommend_detailed(user_id, k)
    return RecommendResponse(
        user_id=user_id,
        items=[RecommendedItem(**item) for item in detailed],
        fallback=not known_to_als,
    )
