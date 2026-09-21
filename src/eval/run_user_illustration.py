"""Standalone re-run of just the side-by-side user illustration.

Split out from run_segment_analysis.py so a bug in this step alone
doesn't require re-running the full (expensive) segment/long-tail
analysis. Fits the same three models the same way.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from src.data.events import load_events  # noqa: E402
from src.data.item_properties import load_category_history  # noqa: E402
from src.data.splits import load_split_config, split_events  # noqa: E402
from src.eval.protocol import eligible_users  # noqa: E402
from src.eval.run_segment_analysis import build_two_stage_model  # noqa: E402
from src.eval.segments import user_training_counts  # noqa: E402
from src.models.als_model import ALSModel  # noqa: E402
from src.models.popularity import PopularityModel  # noqa: E402

ILLUSTRATION_PATH = Path(__file__).resolve().parents[2] / "results" / "user_illustration.json"


def main() -> None:
    events = load_events()
    train, _validation, test = split_events(events)
    split_config = load_split_config()

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

    heavy_user = max(training_counts, key=training_counts.get)
    light_candidates = [u for u, c in training_counts.items() if c == 1]
    if light_candidates:
        light_user = light_candidates[0]
    else:
        light_user = min(training_counts, key=training_counts.get)

    illustration = {
        "heavy_user": {
            "user_id": int(heavy_user),
            "training_count": training_counts[heavy_user],
        },
        "light_user": {
            "user_id": int(light_user),
            "training_count": training_counts[light_user],
        },
        "recommendations": {},
    }

    models = {"popularity": popularity_model, "als": als_model, "two_stage": two_stage_model}
    for model_name, model in models.items():
        illustration["recommendations"][model_name] = {
            "heavy_user_top10": [int(i) for i in model.recommend(heavy_user, 10)],
            "light_user_top10": [int(i) for i in model.recommend(light_user, 10)],
        }

    with ILLUSTRATION_PATH.open("w", encoding="utf-8") as f:
        json.dump(illustration, f, indent=2)

    print(json.dumps(illustration, indent=2))
    print(f"\nWritten to {ILLUSTRATION_PATH}")


if __name__ == "__main__":
    main()
