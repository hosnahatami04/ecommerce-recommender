"""Download and verify the Retailrocket e-commerce dataset.

Primary path: Kaggle CLI (requires a free Kaggle account and an API token
saved at ~/.kaggle/kaggle.json). Manual fallback is documented in the
module docstring below and in the README.

Manual fallback (if the Kaggle CLI path fails, e.g. no API token):
1. Go to https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
2. Click "Download" (requires a free Kaggle login).
3. Unzip the downloaded archive into data/raw/ so that this layout exists:
       data/raw/events.csv
       data/raw/category_tree.csv
       data/raw/item_properties_part1.csv
       data/raw/item_properties_part2.csv
4. Re-run this script with --verify-only to check the files are correct.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "retailrocket/ecommerce-dataset"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# (filename, expected_row_count_min, expected_row_count_max)
# Ranges give slack for minor upstream revisions while still catching a
# truncated download or the wrong file entirely.
EXPECTED_FILES = {
    "events.csv": (2_700_000, 2_760_000),
    "category_tree.csv": (1_600, 1_700),
    "item_properties_part1.csv": (10_000_000, 10_100_000),
    "item_properties_part2.csv": (9_200_000, 9_300_000),
}

EXPECTED_EVENTS_COLUMNS = {"timestamp", "visitorid", "event", "itemid", "transactionid"}


def _count_rows(path: Path) -> int:
    """Count data rows in a CSV (excludes header)."""
    with path.open("rb") as f:
        return sum(1 for _ in f) - 1


def download_via_kaggle() -> bool:
    """Attempt to download the dataset with the kaggle CLI. Returns True on success."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(RAW_DIR),
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[download] Kaggle CLI download failed: {exc}", file=sys.stderr)
        return False

    zip_path = RAW_DIR / "ecommerce-dataset.zip"
    if not zip_path.exists():
        # Kaggle sometimes names the zip after the dataset slug instead.
        candidates = list(RAW_DIR.glob("*.zip"))
        if not candidates:
            print("[download] No zip file found after Kaggle download.", file=sys.stderr)
            return False
        zip_path = candidates[0]

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    return True


def verify() -> bool:
    """Verify that all expected files exist with plausible row counts and schema."""
    ok = True

    for filename, (min_rows, max_rows) in EXPECTED_FILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"[verify] MISSING: {filename}", file=sys.stderr)
            ok = False
            continue

        n_rows = _count_rows(path)
        if not (min_rows <= n_rows <= max_rows):
            print(
                f"[verify] ROW COUNT OUT OF RANGE: {filename} has {n_rows} rows, "
                f"expected {min_rows}-{max_rows}",
                file=sys.stderr,
            )
            ok = False
        else:
            print(f"[verify] OK: {filename} ({n_rows} rows)")

    events_path = RAW_DIR / "events.csv"
    if events_path.exists():
        with events_path.open(encoding="utf-8") as f:
            header = set(f.readline().strip().split(","))
        if header != EXPECTED_EVENTS_COLUMNS:
            print(
                f"[verify] SCHEMA MISMATCH in events.csv: got {header}, "
                f"expected {EXPECTED_EVENTS_COLUMNS}",
                file=sys.stderr,
            )
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download, only verify files already present in data/raw/.",
    )
    args = parser.parse_args()

    if not args.verify_only:
        success = download_via_kaggle()
        if not success:
            print(
                "\n[download] Kaggle CLI download did not complete.\n"
                "Manual fallback:\n"
                "  1. https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset\n"
                "  2. Download and unzip into data/raw/\n"
                "  3. Re-run: python -m src.data.download --verify-only\n",
                file=sys.stderr,
            )
            sys.exit(1)

    if not verify():
        print("\n[verify] Verification FAILED. Dataset is incomplete or corrupt.", file=sys.stderr)
        sys.exit(1)

    print("\n[verify] All files present and verified.")


if __name__ == "__main__":
    main()
