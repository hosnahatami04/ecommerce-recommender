"""Small honest sweep over ALS hyperparameters, selected on validation Recall@10.

The test set is touched exactly once, at the end, with the chosen
config (see run_als.py) — never during this sweep. All sweep results
are saved, including the bad ones, so nothing is cherry-picked after
the fact.
"""

from __future__ import annotations

import os

# Must be set before numpy/implicit are imported anywhere in the process.
# A threaded OpenBLAS fighting implicit's own per-user parallelism causes
# severe slowdowns during ALS fitting.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from src.data.events import load_events
from src.data.splits import split_events
from src.eval.runner import evaluate_model
from src.models.als_model import ALSModel

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "als_sweep.json"

FACTORS_GRID = [32, 64, 128]
REGULARIZATION_GRID = [0.001, 0.01, 0.1]


def main() -> None:
    events = load_events()
    train, validation, _test = split_events(events)
    catalog_size = events["itemid"].nunique()

    sweep_results = []
    for factors in FACTORS_GRID:
        for reg in REGULARIZATION_GRID:
            model = ALSModel(
                factors=factors, regularization=reg, iterations=20, random_state=42
            ).fit(train)
            metrics = evaluate_model(model, train, validation, catalog_size, k_values=(10,))
            entry = {
                "factors": factors,
                "regularization": reg,
                "val_recall_at_10": metrics["recall"]["@10"],
                "val_ndcg_at_10": metrics["ndcg"]["@10"],
            }
            sweep_results.append(entry)
            print(entry)

    best = max(sweep_results, key=lambda r: r["val_recall_at_10"])

    output = {"grid": sweep_results, "best": best}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest config: {best}")
    print(f"Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
