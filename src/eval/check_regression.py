"""CI regression gate: fail if the committed popularity-baseline Recall@10 drops.

This checks the committed results/baseline.json directly rather than
recomputing it, since CI has no access to the raw dataset (it's
gitignored and not fetched in the workflow). If a future PR updates
baseline.json with a real re-run, this floor should be updated too --
the floor exists to catch an accidental regression or corruption of
the file, not to freeze the number forever.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "baseline.json"

# Recorded popularity Recall@10 is ~0.008489. A small tolerance below
# that absorbs floating-point/library-version noise without letting a
# real regression through.
MIN_RECALL_AT_10 = 0.008


def check_regression(results_path: Path) -> tuple[bool, str]:
    """Returns (passed, message)."""
    if not results_path.exists():
        return False, f"[regression-check] MISSING: {results_path}"

    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)

    recall_at_10 = data["popularity"]["recall"]["@10"]

    if recall_at_10 < MIN_RECALL_AT_10:
        return False, (
            f"[regression-check] FAIL: popularity Recall@10 = {recall_at_10:.6f}, "
            f"below the recorded floor of {MIN_RECALL_AT_10}"
        )

    return True, (
        f"[regression-check] OK: popularity Recall@10 = {recall_at_10:.6f} "
        f"(floor: {MIN_RECALL_AT_10})"
    )


def main() -> None:
    passed, message = check_regression(RESULTS_PATH)
    if passed:
        print(message)
    else:
        print(message, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
