"""Record the popularity baseline's numbers on the real dataset.

This is Layer 1. Every later model (ALS, two-stage) is judged as a
delta against results/baseline.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.events import load_events
from src.data.splits import split_events
from src.eval.runner import evaluate_model
from src.models.popularity import PopularityModel, TimeDecayedPopularityModel

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    events = load_events()
    train, validation, test = split_events(events)

    catalog_size = events["itemid"].nunique()

    popularity_model = PopularityModel().fit(train)
    popularity_results = evaluate_model(popularity_model, train, test, catalog_size)

    decayed_model = TimeDecayedPopularityModel(half_life_days=14.0).fit(train)
    decayed_results = evaluate_model(decayed_model, train, test, catalog_size)

    output = {
        "catalog_size": int(catalog_size),
        "n_train_events": int(len(train)),
        "n_test_events": int(len(test)),
        "popularity": popularity_results,
        "popularity_time_decayed": decayed_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
