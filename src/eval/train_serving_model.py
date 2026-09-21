"""Train the two-stage model on all available data and save serving artifacts.

Unlike run_two_stage.py (which holds out the test set for evaluation),
this trains on train + validation + test combined, since the resulting
artifacts are what actually serves live traffic and should use every
real event available. Evaluation numbers reported in the README always
come from the held-out runs in results/, never from this model.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from src.data.events import load_events  # noqa: E402
from src.data.item_properties import category_as_of, load_category_history  # noqa: E402
from src.data.splits import (  # noqa: E402
    load_split_config,
    split_events,
    split_train_for_reranker,
)
from src.models.als_model import ALSModel  # noqa: E402
from src.models.artifacts import save_artifacts  # noqa: E402
from src.models.features import FeatureContext, build_features  # noqa: E402
from src.models.popularity import PopularityModel  # noqa: E402
from src.models.reranker import LightGBMReranker  # noqa: E402
from src.models.reranker_labels import build_labeled_candidates  # noqa: E402

POOL_SIZE = 200


def main() -> None:
    events = load_events()
    train, validation, test = split_events(events)
    split_config = load_split_config()
    all_events = events  # full dataset, used for the serving models

    print("Building reranker training labels (leakage-safe, train-window only)...")
    sub_train, label_window = split_train_for_reranker(train, split_config)
    label_als = ALSModel().fit(sub_train)
    label_popularity = PopularityModel().fit(sub_train)
    labeled = build_labeled_candidates(
        sub_train,
        label_window,
        pool_size=POOL_SIZE,
        als_model=label_als,
        popularity_model=label_popularity,
    )

    print("Loading category history...")
    category_history = load_category_history()
    label_window_start_ms = split_config["reranker_label_start_ms"]
    label_category_map = category_as_of(category_history, cutoff_ms=label_window_start_ms)
    label_ctx = FeatureContext(
        sub_train, cutoff_ms=label_window_start_ms, category_as_of=label_category_map
    )
    labeled_features = build_features(labeled, label_ctx, label_als)
    labeled_features["label"] = labeled["label"]

    print("Fitting reranker...")
    reranker = LightGBMReranker(random_state=42, n_estimators=100).fit(labeled_features)

    print("Fitting serving ALS + popularity on the full dataset...")
    serving_als = ALSModel(
        factors=128, regularization=0.01, iterations=20, random_state=42
    ).fit(all_events)
    serving_popularity = PopularityModel().fit(all_events)

    serving_cutoff_ms = int(all_events["timestamp"].max()) + 1
    serving_category_map = category_as_of(category_history, cutoff_ms=serving_cutoff_ms)
    serving_ctx = FeatureContext(
        all_events, cutoff_ms=serving_cutoff_ms, category_as_of=serving_category_map
    )

    print("Saving artifacts...")
    save_artifacts(serving_als, serving_popularity, reranker, serving_ctx)
    print("Done.")


if __name__ == "__main__":
    main()
