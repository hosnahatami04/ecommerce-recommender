import functools
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.als_model import ALSModel
from src.models.artifacts import load_artifacts, save_artifacts
from src.models.features import FeatureContext, build_features
from src.models.popularity import PopularityModel
from src.models.reranker import LightGBMReranker


def _build_fixture_artifacts(artifacts_dir: Path) -> None:
    """Small but real trained artifacts -- known user 1, unknown user 999."""
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 2, 3],
            "itemid": [10, 20, 10, 30, 20],
            "timestamp": [1, 2, 3, 4, 5],
            "event": ["view", "addtocart", "view", "transaction", "view"],
        }
    )
    als_model = ALSModel(factors=2, iterations=3).fit(train)
    popularity_model = PopularityModel().fit(train)
    ctx = FeatureContext(train, cutoff_ms=100)

    pairs = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 20, 30]})
    featured = build_features(pairs, ctx, als_model)
    featured["label"] = [1, 0, 1]
    reranker = LightGBMReranker(random_state=42, n_estimators=5).fit(featured)

    save_artifacts(als_model, popularity_model, reranker, ctx, artifacts_dir=artifacts_dir)


@pytest.fixture
def client(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    _build_fixture_artifacts(artifacts_dir)

    monkeypatch.setattr(
        "src.api.load_artifacts", functools.partial(load_artifacts, artifacts_dir=artifacts_dir)
    )

    from src.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok_when_artifacts_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_known_user_returns_items(client) -> None:
    response = client.get("/recommend/1?k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["fallback"] is False
    assert len(body["items"]) <= 5
    for item in body["items"]:
        assert "item_id" in item
        assert "score" in item
        assert item["source"] in {"als", "popularity_fallback"}
        assert isinstance(item["rerank_moved"], bool)


def test_recommend_unknown_user_returns_fallback_true(client) -> None:
    response = client.get("/recommend/999999?k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is True
    # An unknown user must still get real recommendations, not an error.
    assert len(body["items"]) > 0


def test_recommend_respects_k_parameter(client) -> None:
    response = client.get("/recommend/1?k=2")
    assert response.status_code == 200
    assert len(response.json()["items"]) <= 2


def test_recommend_rejects_invalid_k(client) -> None:
    response = client.get("/recommend/1?k=0")
    assert response.status_code == 422

    response = client.get("/recommend/1?k=1000")
    assert response.status_code == 422
