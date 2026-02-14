from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import (
    DEFAULT_OPTIONAL_COLUMNS,
    DEFAULT_REQUIRED_COLUMNS,
    load_schema_contracts,
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
