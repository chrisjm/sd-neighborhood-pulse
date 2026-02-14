from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, promote_latest, validate_columns


def _write_csv(path: Path, headers: list[str], row: list[str]) -> None:
    path.write_text(",".join(headers) + "\n" + ",".join(row) + "\n", encoding="utf-8")


def test_validate_columns_allows_missing_optional_field(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    headers = REQUIRED_COLUMNS.copy()
    row = ["x"] * len(headers)
    _write_csv(csv_path, headers, row)

    result = validate_columns(csv_path, REQUIRED_COLUMNS, OPTIONAL_COLUMNS)

    assert result["row_count_sample"] == 1
    assert result["missing_optional"] == ["specify_the_issue"]


def test_validate_columns_raises_when_required_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_required.csv"
    headers = [column for column in REQUIRED_COLUMNS if column != "service_name"]
    row = ["x"] * len(headers)
    _write_csv(csv_path, headers, row)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(csv_path, REQUIRED_COLUMNS, OPTIONAL_COLUMNS)


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
