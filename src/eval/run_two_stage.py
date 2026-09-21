"""Train the full two-stage model (ALS retrieve + LightGBM rerank) and record results.

Pipeline:
1. Split train into sub_train / label_window (both strictly inside the
   original train period).
2. Fit ALS + popularity on sub_train; generate labeled candidates for
   label_window users.
3. Build features for every labeled candidate, using only data visible
   before the label window starts.
4. Fit the LightGBM reranker on those labeled features.
5. Re-fit ALS + popularity on the FULL train set (not just sub_train)
   for serving -- the reranker itself doesn't need to see test data,
   but the retrieval stage should use all available training history.
6. Evaluate the full two-stage model on the test set, exactly once.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from src.data.events import load_events  # noqa: E402
from src.data.item_properties import category_as_of, load_category_history  # noqa: E402
from src.data.splits import (  # noqa: E402
    load_split_config,
    split_events,
    split_train_for_reranker,
)
from src.eval.runner import evaluate_model  # noqa: E402
from src.models.als_model import ALSModel  # noqa: E402
from src.models.features import FeatureContext, build_features  # noqa: E402
from src.models.popularity import PopularityModel  # noqa: E402
from src.models.reranker import LightGBMReranker  # noqa: E402
from src.models.reranker_labels import build_labeled_candidates  # noqa: E402
from src.models.two_stage import TwoStageModel  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "two_stage.json"
POOL_SIZE = 200


def main() -> None:
    events = load_events()
    train, _validation, test = split_events(events)
    split_config = load_split_config()
    catalog_size = events["itemid"].nunique()

    sub_train, label_window = split_train_for_reranker(train, split_config)
    print(f"sub_train: {len(sub_train)} events, label_window: {len(label_window)} events")

    label_als = ALSModel().fit(sub_train)
    label_popularity = PopularityModel().fit(sub_train)

    print("Building labeled candidates...")
    labeled = build_labeled_candidates(
        sub_train,
        label_window,
        pool_size=POOL_SIZE,
        als_model=label_als,
        popularity_model=label_popularity,
    )
    positive_rate = labeled["label"].mean()
    print(f"Labeled candidates: {len(labeled)} rows, positive rate: {positive_rate:.4%}")

    print("Loading category history...")
    category_history = load_category_history()
    label_window_start_ms = split_config["reranker_label_start_ms"]
    label_category_map = category_as_of(category_history, cutoff_ms=label_window_start_ms)

    label_ctx = FeatureContext(
        sub_train, cutoff_ms=label_window_start_ms, category_as_of=label_category_map
    )
    print("Building features for labeled candidates...")
    labeled_features = build_features(labeled, label_ctx, label_als)
    labeled_features["label"] = labeled["label"]

    print("Fitting reranker...")
    reranker = LightGBMReranker(random_state=42, n_estimators=100)
    reranker.fit(labeled_features)

    print("Re-fitting ALS + popularity on full train for serving...")
    serving_als = ALSModel().fit(train)
    serving_popularity = PopularityModel().fit(train)

    val_start_ms = split_config["val_start_ms"]
    serving_category_map = category_as_of(category_history, cutoff_ms=val_start_ms)
    serving_ctx = FeatureContext(train, cutoff_ms=val_start_ms, category_as_of=serving_category_map)

    two_stage_model = TwoStageModel(
        serving_als, serving_popularity, reranker, serving_ctx, pool_size=POOL_SIZE
    )

    print("Evaluating on test set (this is the only test-set touch)...")
    results = evaluate_model(
        two_stage_model, train, test, catalog_size, candidate_pool_size=POOL_SIZE
    )

    output = {
        "pool_size": POOL_SIZE,
        "labeled_candidates_positive_rate": float(labeled["label"].mean()),
        "catalog_size": int(catalog_size),
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
