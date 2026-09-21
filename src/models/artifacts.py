"""Save and load trained model artifacts for serving.

Artifacts are saved with a version stamp (the git-independent training
timestamp) so a stale or missing artifact directory fails loudly at
load time instead of silently serving whatever happened to be on disk.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

MANIFEST_FILENAME = "manifest.json"
ALS_FILENAME = "als_model.pkl"
POPULARITY_FILENAME = "popularity_model.pkl"
RERANKER_FILENAME = "reranker.pkl"
FEATURE_CONTEXT_FILENAME = "feature_context.pkl"


def save_artifacts(
    als_model,
    popularity_model,
    reranker,
    feature_context,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / ALS_FILENAME).open("wb") as f:
        pickle.dump(als_model, f)
    with (artifacts_dir / POPULARITY_FILENAME).open("wb") as f:
        pickle.dump(popularity_model, f)
    with (artifacts_dir / RERANKER_FILENAME).open("wb") as f:
        pickle.dump(reranker, f)
    with (artifacts_dir / FEATURE_CONTEXT_FILENAME).open("wb") as f:
        pickle.dump(feature_context, f)

    manifest = {
        "trained_at": datetime.now(UTC).isoformat(),
        "files": [ALS_FILENAME, POPULARITY_FILENAME, RERANKER_FILENAME, FEATURE_CONTEXT_FILENAME],
    }
    with (artifacts_dir / MANIFEST_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> tuple:
    """Load (als_model, popularity_model, reranker, feature_context).

    Fails loudly (FileNotFoundError with a clear message) if the
    manifest or any referenced file is missing -- a serving process
    with stale or absent artifacts should refuse to start, not silently
    serve garbage.
    """
    manifest_path = artifacts_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No model manifest found at {manifest_path}. "
            "Train and save artifacts before starting the API "
            "(see src/eval/run_segment_analysis.py or a dedicated training script)."
        )

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    for filename in manifest["files"]:
        path = artifacts_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest references {filename} but it is missing at {path}. "
                "Artifacts directory is incomplete or corrupt -- retrain and re-save."
            )

    with (artifacts_dir / ALS_FILENAME).open("rb") as f:
        als_model = pickle.load(f)
    with (artifacts_dir / POPULARITY_FILENAME).open("rb") as f:
        popularity_model = pickle.load(f)
    with (artifacts_dir / RERANKER_FILENAME).open("rb") as f:
        reranker = pickle.load(f)
    with (artifacts_dir / FEATURE_CONTEXT_FILENAME).open("rb") as f:
        feature_context = pickle.load(f)

    return als_model, popularity_model, reranker, feature_context
