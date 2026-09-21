import json
from pathlib import Path

from src.eval.check_regression import check_regression


def _write_baseline(path: Path, recall_at_10: float) -> None:
    data = {"popularity": {"recall": {"@10": recall_at_10}}}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def test_check_regression_passes_above_floor(tmp_path: Path) -> None:
    results_path = tmp_path / "baseline.json"
    _write_baseline(results_path, recall_at_10=0.01)
    passed, message = check_regression(results_path)
    assert passed is True
    assert "OK" in message


def test_check_regression_fails_below_floor(tmp_path: Path) -> None:
    results_path = tmp_path / "baseline.json"
    _write_baseline(results_path, recall_at_10=0.001)
    passed, message = check_regression(results_path)
    assert passed is False
    assert "FAIL" in message


def test_check_regression_fails_on_missing_file(tmp_path: Path) -> None:
    results_path = tmp_path / "does_not_exist.json"
    passed, message = check_regression(results_path)
    assert passed is False
    assert "MISSING" in message


def test_check_regression_passes_on_real_committed_baseline() -> None:
    real_path = (
        Path(__file__).resolve().parents[1] / "results" / "baseline.json"
    )
    passed, _message = check_regression(real_path)
    assert passed is True
