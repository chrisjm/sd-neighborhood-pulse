from pathlib import Path
import logging
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import (
    DEFAULT_OPTIONAL_COLUMNS,
    DEFAULT_REQUIRED_COLUMNS,
    is_backfill_once_source,
    is_daily_refresh_source,
    load_schema_contracts,
    process_source,
    promote_latest,
    validate_columns,
)


def _write_csv(path: Path, headers: list[str], row: list[str]) -> None:
    path.write_text(",".join(headers) + "\n" + ",".join(row) + "\n", encoding="utf-8")


def test_validate_columns_allows_missing_optional_field(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    headers = DEFAULT_REQUIRED_COLUMNS.copy()
    row = ["x"] * len(headers)
    _write_csv(csv_path, headers, row)

    result = validate_columns(csv_path, DEFAULT_REQUIRED_COLUMNS, DEFAULT_OPTIONAL_COLUMNS)

    assert result["row_count_sample"] == 1
    assert result["missing_optional"] == ["specify_the_issue"]


def test_validate_columns_raises_when_required_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_required.csv"
    headers = [column for column in DEFAULT_REQUIRED_COLUMNS if column != "service_name"]
    row = ["x"] * len(headers)
    _write_csv(csv_path, headers, row)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(csv_path, DEFAULT_REQUIRED_COLUMNS, DEFAULT_OPTIONAL_COLUMNS)


def test_load_schema_contracts_reads_required_and_optional(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dictionary.csv"
    dictionary_path.write_text(
        "field,description,data_type\n"
        "service_request_id,id,String\n"
        "service_name,name,String\n"
        "specify_the_issue,optional,String\n",
        encoding="utf-8",
    )

    required, optional = load_schema_contracts(dictionary_path=dictionary_path)

    assert required == ["service_request_id", "service_name"]
    assert optional == ["specify_the_issue"]


def test_load_schema_contracts_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schema_contracts(dictionary_path=tmp_path / "missing.csv")


def test_promote_latest_copies_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.csv"
    latest_path = tmp_path / "latest.csv"
    archive_path.write_text("a,b\n1,2\n", encoding="utf-8")

    promote_latest(archive_path, latest_path, dry_run=False)

    assert latest_path.exists()
    assert latest_path.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_promote_latest_dry_run_does_not_copy(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.csv"
    latest_path = tmp_path / "latest.csv"
    archive_path.write_text("a,b\n1,2\n", encoding="utf-8")

    promote_latest(archive_path, latest_path, dry_run=True)

    assert not latest_path.exists()


def test_source_classification_matches_refresh_policy() -> None:
    assert is_daily_refresh_source("open_current")
    assert is_daily_refresh_source("closed_2026")
    assert not is_daily_refresh_source("closed_2025")
    assert is_backfill_once_source("closed_2025")


def test_process_source_skips_backfill_when_latest_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.csv"
    latest_path = tmp_path / "latest.csv"
    latest_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def fake_build_paths(source_key: str, run_date: str) -> tuple[Path, Path]:
        return archive_path, latest_path

    monkeypatch.setattr("scripts.ingest.build_paths", fake_build_paths)
    monkeypatch.setattr(
        "scripts.ingest.validate_columns",
        lambda path, required, optional: {"row_count_sample": 1, "missing_optional": []},
    )

    calls = {"download": 0, "promote": 0}

    def fake_download(url: str, destination: Path) -> None:
        calls["download"] += 1

    def fake_promote(source: Path, destination: Path, dry_run: bool = False) -> None:
        calls["promote"] += 1

    monkeypatch.setattr("scripts.ingest.download_csv", fake_download)
    monkeypatch.setattr("scripts.ingest.promote_latest", fake_promote)

    logger = logging.getLogger("test-ingest-backfill")
    result = process_source(
        source_key="closed_2025",
        url="https://example.test/closed_2025.csv",
        run_date="2026-02-14",
        force=False,
        dry_run=False,
        required_columns=DEFAULT_REQUIRED_COLUMNS,
        optional_columns=DEFAULT_OPTIONAL_COLUMNS,
        logger=logger,
    )

    assert result is True
    assert calls["download"] == 0
    assert calls["promote"] == 0


def test_process_source_daily_refresh_skips_when_archive_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.csv"
    latest_path = tmp_path / "latest.csv"
    archive_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def fake_build_paths(source_key: str, run_date: str) -> tuple[Path, Path]:
        return archive_path, latest_path

    monkeypatch.setattr("scripts.ingest.build_paths", fake_build_paths)
    monkeypatch.setattr(
        "scripts.ingest.validate_columns",
        lambda path, required, optional: {"row_count_sample": 1, "missing_optional": []},
    )

    calls = {"download": 0, "promote": 0}

    def fake_download(url: str, destination: Path) -> None:
        calls["download"] += 1

    def fake_promote(source: Path, destination: Path, dry_run: bool = False) -> None:
        calls["promote"] += 1

    monkeypatch.setattr("scripts.ingest.download_csv", fake_download)
    monkeypatch.setattr("scripts.ingest.promote_latest", fake_promote)

    logger = logging.getLogger("test-ingest-daily")
    result = process_source(
        source_key="open_current",
        url="https://example.test/open_current.csv",
        run_date="2026-02-14",
        force=False,
        dry_run=False,
        required_columns=DEFAULT_REQUIRED_COLUMNS,
        optional_columns=DEFAULT_OPTIONAL_COLUMNS,
        logger=logger,
    )

    assert result is True
    assert calls["download"] == 0
    assert calls["promote"] == 1
