"""p50 / p95 latency benchmark for the /recommend endpoint on CPU.

Runs against the real trained artifacts (not a fixture), driving the
FastAPI app in-process via TestClient to avoid uvicorn startup/network
overhead skewing the numbers -- what's measured is model inference
cost, not HTTP transport.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fastapi.testclient import TestClient  # noqa: E402

from src.data.events import load_events  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "latency.json"
N_REQUESTS = 200


def main() -> None:
    from src.api import app

    events = load_events()
    sample_users = events["visitorid"].drop_duplicates().sample(
        n=min(N_REQUESTS, events["visitorid"].nunique()), random_state=42
    ).tolist()

    with TestClient(app) as client:
        latencies_ms = []
        for user_id in sample_users:
            start = time.perf_counter()
            response = client.get(f"/recommend/{user_id}?k=10")
            elapsed_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[int(n * 0.50)]
    p95 = latencies_ms[int(n * 0.95)]

    output = {
        "n_requests": n,
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
