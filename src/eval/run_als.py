"""Train the final ALS config and record results against the test set.

The config here should be the winner from results/als_sweep.json's
validation-selected "best" entry. The test set is touched exactly once
here, at the very end — never during the sweep.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from src.data.events import load_events  # noqa: E402
from src.data.splits import split_events  # noqa: E402
from src.eval.runner import evaluate_model  # noqa: E402
from src.models.als_model import ALSModel  # noqa: E402

SWEEP_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "als_sweep.json"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results" / "als.json"


def main() -> None:
    with SWEEP_RESULTS_PATH.open(encoding="utf-8") as f:
        sweep = json.load(f)
    best = sweep["best"]

    events = load_events()
    train, _validation, test = split_events(events)
    catalog_size = events["itemid"].nunique()

    model = ALSModel(
        factors=best["factors"],
        regularization=best["regularization"],
        iterations=20,
        random_state=42,
    ).fit(train)

    results = evaluate_model(model, train, test, catalog_size)

    output = {
        "chosen_config": {
            "factors": best["factors"],
            "regularization": best["regularization"],
            "iterations": 20,
            "random_state": 42,
        },
        "selected_on": "validation Recall@10",
        "catalog_size": int(catalog_size),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
