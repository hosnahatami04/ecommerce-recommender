"""Cold-start, coverage, and popularity-bias analysis across user segments.

Runs the popularity baseline, ALS, and the two-stage model through the
identical protocol, broken out by training-history segment (0, 1-2,
3-10, 11+ events). Averages hide failure -- this is where it gets
un-hidden.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from src.data.events import load_events  # noqa: E402
from src.data.item_properties import category_as_of, load_category_history  # noqa: E402
from src.data.splits import load_split_config, split_events  # noqa: E402
from src.eval.longtail import long_tail_table  # noqa: E402
from src.eval.protocol import eligible_users  # noqa: E402
from src.eval.runner import evaluate_model  # noqa: E402
from src.eval.segments import segment_users, user_training_counts  # noqa: E402
from src.models.als_model import ALSModel  # noqa: E402
from src.models.features import FeatureContext  # noqa: E402
from src.models.popularity import PopularityModel  # noqa: E402
from src.models.reranker import LightGBMReranker  # noqa: E402
from src.models.reranker_labels import build_labeled_candidates  # noqa: E402
from src.models.two_stage import TwoStageModel  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "segment_analysis.json"
LONGTAIL_PATH = Path(__file__).resolve().parents[2] / "results" / "longtail.json"
ILLUSTRATION_PATH = Path(__file__).resolve().parents[2] / "results" / "user_illustration.json"
POOL_SIZE = 200


def build_two_stage_model(train, split_config, category_history) -> TwoStageModel:
    from src.data.splits import split_train_for_reranker
    from src.models.features import build_features

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
    label_window_start_ms = split_config["reranker_label_start_ms"]
    label_category_map = category_as_of(category_history, cutoff_ms=label_window_start_ms)
    label_ctx = FeatureContext(
        sub_train, cutoff_ms=label_window_start_ms, category_as_of=label_category_map
    )
    labeled_features = build_features(labeled, label_ctx, label_als)
    labeled_features["label"] = labeled["label"]

    reranker = LightGBMReranker(random_state=42, n_estimators=100).fit(labeled_features)

    serving_als = ALSModel().fit(train)
    serving_popularity = PopularityModel().fit(train)
    val_start_ms = split_config["val_start_ms"]
    serving_category_map = category_as_of(category_history, cutoff_ms=val_start_ms)
    serving_ctx = FeatureContext(
        train, cutoff_ms=val_start_ms, category_as_of=serving_category_map
    )

    return TwoStageModel(
        serving_als, serving_popularity, reranker, serving_ctx, pool_size=POOL_SIZE
    )


def main() -> None:
    events = load_events()
    train, _validation, test = split_events(events)
    split_config = load_split_config()
    catalog_size = events["itemid"].nunique()

    print("Fitting popularity baseline...")
    popularity_model = PopularityModel().fit(train)

    print("Fitting ALS...")
    als_model = ALSModel(
        factors=128, regularization=0.01, iterations=20, random_state=42
    ).fit(train)

    print("Loading category history...")
    category_history = load_category_history()

    print("Building two-stage model...")
    two_stage_model = build_two_stage_model(train, split_config, category_history)

    all_test_users = eligible_users(test)
    training_counts = user_training_counts(train, all_test_users)
    segments = segment_users(training_counts)

    for label, users in segments.items():
        print(f"segment {label}: {len(users)} users")

    models = {
        "popularity": popularity_model,
        "als": als_model,
        "two_stage": two_stage_model,
    }

    segment_results: dict[str, dict] = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name} per segment...")
        segment_results[model_name] = {}
        for label, users in segments.items():
            if not users:
                segment_results[model_name][label] = None
                continue
            results = evaluate_model(
                model, train, test, catalog_size, candidate_pool_size=POOL_SIZE, user_subset=users
            )
            segment_results[model_name][label] = results

    output = {
        "segment_sizes": {label: len(users) for label, users in segments.items()},
        "results_by_model": segment_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written to {RESULTS_PATH}")

    # Long-tail analysis: full-population recommendations for each model.
    print("Building long-tail tables...")
    all_users = sorted(all_test_users)
    longtail_output = {}
    for model_name, model in models.items():
        all_recs = [model.recommend(u, POOL_SIZE) for u in all_users]
        table = long_tail_table(all_recs, train, k=10)
        top_ranked = table.sort_values("popularity_rank")
        longtail_output[model_name] = {
            "popularity_rank": top_ranked["popularity_rank"].tolist(),
            "recommend_count": top_ranked["recommend_count"].tolist(),
        }
    with LONGTAIL_PATH.open("w", encoding="utf-8") as f:
        json.dump(longtail_output, f)
    print(f"Written to {LONGTAIL_PATH}")

    # Side-by-side illustration: one heavy user, one light user.
    print("Building user illustration...")
    heavy_user = max(training_counts, key=training_counts.get)
    light_candidates = [u for u, c in training_counts.items() if c == 1]
    if light_candidates:
        light_user = light_candidates[0]
    else:
        light_user = min(training_counts, key=training_counts.get)

    illustration = {
        "heavy_user": {"user_id": int(heavy_user), "training_count": training_counts[heavy_user]},
        "light_user": {"user_id": int(light_user), "training_count": training_counts[light_user]},
        "recommendations": {},
    }
    for model_name, model in models.items():
        illustration["recommendations"][model_name] = {
            "heavy_user_top10": [int(i) for i in model.recommend(heavy_user, 10)],
            "light_user_top10": [int(i) for i in model.recommend(light_user, 10)],
        }
    with ILLUSTRATION_PATH.open("w", encoding="utf-8") as f:
        json.dump(illustration, f, indent=2)
    print(f"Written to {ILLUSTRATION_PATH}")


if __name__ == "__main__":
    main()
