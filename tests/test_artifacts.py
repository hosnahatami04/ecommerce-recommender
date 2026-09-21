from pathlib import Path

import pandas as pd
import pytest

from src.models.als_model import ALSModel
from src.models.artifacts import load_artifacts, save_artifacts
from src.models.features import FeatureContext
from src.models.popularity import PopularityModel
from src.models.reranker import LightGBMReranker


def _toy_trained_pieces():
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 2],
            "itemid": [10, 20, 10],
            "timestamp": [1, 2, 3],
            "event": ["view", "view", "view"],
        }
    )
    als_model = ALSModel(factors=2, iterations=2).fit(train)
    popularity_model = PopularityModel().fit(train)
    ctx = FeatureContext(train, cutoff_ms=10)

    labeled = pd.DataFrame(
        {
            "visitorid": [1, 1],
            "itemid": [10, 20],
            "als_score": [0.1, 0.2],
            "item_popularity": [0.5, 0.5],
            "item_popularity_trend": [0.1, 0.1],
            "item_age_days": [1.0, 1.0],
            "user_activity": [2, 2],
            "category_affinity_match": [0, 0],
            "days_since_last_event": [1.0, 1.0],
            "label": [1, 0],
        }
    )
    reranker = LightGBMReranker(random_state=42, n_estimators=5).fit(labeled)

    return als_model, popularity_model, reranker, ctx


def test_save_and_load_artifacts_round_trip(tmp_path: Path) -> None:
    als_model, popularity_model, reranker, ctx = _toy_trained_pieces()
    save_artifacts(als_model, popularity_model, reranker, ctx, artifacts_dir=tmp_path)

    loaded_als, loaded_pop, loaded_rr, loaded_ctx = load_artifacts(artifacts_dir=tmp_path)

    assert loaded_als.recommend(1, 5) == als_model.recommend(1, 5)
    assert loaded_pop.recommend(1, 5) == popularity_model.recommend(1, 5)


def test_load_artifacts_missing_manifest_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No model manifest found"):
        load_artifacts(artifacts_dir=tmp_path)


def test_load_artifacts_missing_file_fails_loudly(tmp_path: Path) -> None:
    als_model, popularity_model, reranker, ctx = _toy_trained_pieces()
    save_artifacts(als_model, popularity_model, reranker, ctx, artifacts_dir=tmp_path)

    (tmp_path / "als_model.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="Artifacts directory is incomplete"):
        load_artifacts(artifacts_dir=tmp_path)


def test_save_artifacts_writes_manifest(tmp_path: Path) -> None:
    als_model, popularity_model, reranker, ctx = _toy_trained_pieces()
    save_artifacts(als_model, popularity_model, reranker, ctx, artifacts_dir=tmp_path)
    assert (tmp_path / "manifest.json").exists()
