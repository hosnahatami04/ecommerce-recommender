"""Compute and print corpus statistics from the real event log.

These numbers (event counts by type, unique users/items, matrix
sparsity, events-per-user distribution) drive later modeling decisions
and are quoted in the final README — they are always measured here,
never quoted from memory or the internet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.events import load_events

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "corpus_stats.json"


def compute_corpus_stats(events: pd.DataFrame) -> dict:
    n_events = len(events)
    n_users = events["visitorid"].nunique()
    n_items = events["itemid"].nunique()
    sparsity_pct = n_events / (n_users * n_items) * 100

    events_per_user = events.groupby("visitorid").size()

    return {
        "n_events": int(n_events),
        "events_by_type": events["event"].value_counts().to_dict(),
        "n_unique_users": int(n_users),
        "n_unique_items": int(n_items),
        "matrix_sparsity_pct": sparsity_pct,
        "events_per_user": {
            "mean": float(events_per_user.mean()),
            "median": float(events_per_user.median()),
            "min": int(events_per_user.min()),
            "max": int(events_per_user.max()),
            "p25": float(events_per_user.quantile(0.25)),
            "p75": float(events_per_user.quantile(0.75)),
            "n_users_with_1_event": int((events_per_user == 1).sum()),
            "n_users_with_3plus_events": int((events_per_user >= 3).sum()),
        },
        "timestamp_min": int(events["timestamp"].min()),
        "timestamp_max": int(events["timestamp"].max()),
    }


def main() -> None:
    events = load_events()
    stats = compute_corpus_stats(events)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
