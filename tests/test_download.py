from pathlib import Path

from src.data import download


def test_count_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    assert download._count_rows(csv_path) == 3


def test_count_rows_header_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("a,b\n", encoding="utf-8")
    assert download._count_rows(csv_path) == 0


def test_verify_reports_missing_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    assert download.verify() is False


def test_verify_passes_with_valid_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        download,
        "EXPECTED_FILES",
        {"events.csv": (2, 4), "category_tree.csv": (1, 2)},
    )

    events = tmp_path / "events.csv"
    events.write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "1,1,view,10,\n"
        "2,1,view,11,\n"
        "3,2,addtocart,10,\n",
        encoding="utf-8",
    )
    (tmp_path / "category_tree.csv").write_text("categoryid,parentid\n1,\n", encoding="utf-8")

    assert download.verify() is True


def test_verify_flags_schema_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download, "EXPECTED_FILES", {"events.csv": (1, 3)})

    events = tmp_path / "events.csv"
    events.write_text("timestamp,visitorid,event\n1,1,view\n2,1,view\n", encoding="utf-8")

    assert download.verify() is False
