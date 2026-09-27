"""Small honest sweep over event confidence weights, selected on validation Recall@10.

The committed default (view=1, addtocart=3, transaction=5) was a
starting assumption from Phase 1, never re-tested. This sweeps a few
meaningfully different weighting schemes -- using the best config from
the Phase 3 ALS sweep (factors=128, regularization=0.01) so only the
weights vary -- and picks a winner on validation only. The test set is
touched exactly once, at the end, with the chosen config.
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

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "weights_sweep.json"

# Each scheme is a meaningfully different hypothesis, not a small
# perturbation of the default:
WEIGHT_SCHEMES = {
    "default (1/3/5)": {"view": 1, "addtocart": 3, "transaction": 5},
    "purchase-heavy (1/3/10)": {"view": 1, "addtocart": 3, "transaction": 10},
    "cart-and-purchase-heavy (1/5/10)": {"view": 1, "addtocart": 5, "transaction": 10},
    "purchase-only (0/0/1)": {"view": 0, "addtocart": 0, "transaction": 1},
    "log-scaled-ish (1/2/3)": {"view": 1, "addtocart": 2, "transaction": 3},
}

FACTORS = 128
REGULARIZATION = 0.01


def main() -> None:
    events = load_events()
    train, validation, _test = split_events(events)
    catalog_size = events["itemid"].nunique()

    sweep_results = []
    for name, weights in WEIGHT_SCHEMES.items():
        # purchase-only assigns weight 0 to view/addtocart -- those
        # rows would contribute nothing, so drop them before fitting
        # rather than build a matrix with zero-weight cells.
        effective_weights = {k: v for k, v in weights.items() if v > 0}
        train_for_scheme = train[train["event"].isin(effective_weights.keys())]

        model = ALSModel(
            factors=FACTORS, regularization=REGULARIZATION, iterations=20, random_state=42
        ).fit(train_for_scheme, weights=effective_weights)

        metrics = evaluate_model(model, train, validation, catalog_size, k_values=(10,))
        entry = {
            "name": name,
            "weights": weights,
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

    print(f"\nBest scheme: {best}")
    print(f"Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
