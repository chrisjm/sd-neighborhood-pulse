import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

SOURCE_REGISTRY = {
    "open_current": "https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_open_datasd.csv",
    "closed_2026": "https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_closed_2026_datasd.csv",
    "closed_2025": "https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_closed_2025_datasd.csv",
}

REQUIRED_COLUMNS = [
    "service_request_id",
    "service_request_parent_id",
    "sap_notification_number",
    "date_requested",
    "case_age_days",
    "case_record_type",
    "service_name",
    "service_name_detail",
    "date_closed",
    "status",
    "lat",
    "lng",
    "street_address",
    "zipcode",
    "council_district",
    "comm_plan_code",
    "comm_plan_name",
    "park_name",
    "case_origin",
    "referred",
    "iamfloc",
    "floc",
    "public_description",
]
OPTIONAL_COLUMNS = ["specify_the_issue"]


def configure_logging() -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ingest")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(logs_dir / "ingest.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_paths(source_key: str, run_date: str) -> tuple[Path, Path]:
    archive_dir = Path("data/raw/archive") / source_key
    latest_dir = Path("data/raw/latest")
    archive_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / f"get_it_done_{source_key}_{run_date}.csv"
    latest_path = latest_dir / f"get_it_done_{source_key}_latest.csv"
    return archive_path, latest_path


def download_csv(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)


def validate_columns(path: Path, required: list[str], optional: list[str]) -> dict:
    sample = pd.read_csv(path, nrows=100)
    if sample.empty:
        raise ValueError("downloaded file is empty")

    missing_required = [column for column in required if column not in sample.columns]
    if missing_required:
        raise ValueError(f"missing required columns: {missing_required}")

    missing_optional = [column for column in optional if column not in sample.columns]
    return {
        "row_count_sample": len(sample),
        "missing_optional": missing_optional,
    }


def promote_latest(archive_path: Path, latest_path: Path, dry_run: bool = False) -> None:
    if dry_run:
        return
    shutil.copy2(archive_path, latest_path)


def process_source(
    source_key: str,
    url: str,
    run_date: str,
    force: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> bool:
    archive_path, latest_path = build_paths(source_key, run_date)
    logger.info("source=%s step=start url=%s", source_key, url)

    try:
        if dry_run and not archive_path.exists():
            logger.info(
                "source=%s step=dry_run_skip reason=no_archive_present path=%s",
                source_key,
                archive_path,
            )
            return True

        if archive_path.exists() and not force:
            validation = validate_columns(archive_path, REQUIRED_COLUMNS, OPTIONAL_COLUMNS)
            logger.info(
                "source=%s step=skip_download reason=archive_exists sample_rows=%s missing_optional=%s",
                source_key,
                validation["row_count_sample"],
                validation["missing_optional"],
            )
        else:
            if dry_run:
                logger.info("source=%s step=dry_run_download path=%s", source_key, archive_path)
            else:
                download_csv(url, archive_path)
                logger.info("source=%s step=download_complete path=%s", source_key, archive_path)

            validation = validate_columns(archive_path, REQUIRED_COLUMNS, OPTIONAL_COLUMNS)
            logger.info(
                "source=%s step=validated sample_rows=%s missing_optional=%s",
                source_key,
                validation["row_count_sample"],
                validation["missing_optional"],
            )

        promote_latest(archive_path, latest_path, dry_run=dry_run)
        logger.info("source=%s step=latest_updated path=%s dry_run=%s", source_key, latest_path, dry_run)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("source=%s step=failed error=%s", source_key, exc)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest San Diego Get It Done datasets")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", *SOURCE_REGISTRY.keys()],
        help="Dataset source key to ingest",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Run date used in archive file naming (YYYY-MM-DD)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if archive exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate workflow without writing files")
    return parser.parse_args()


def ingest_data() -> int:
    args = parse_args()
    logger = configure_logging()

    if args.source == "all":
        selected_sources = list(SOURCE_REGISTRY.items())
    else:
        selected_sources = [(args.source, SOURCE_REGISTRY[args.source])]

    failures = []
    for source_key, url in selected_sources:
        success = process_source(
            source_key=source_key,
            url=url,
            run_date=args.date,
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )
        if not success:
            failures.append(source_key)

    if failures:
        logger.error("run_status=failed failed_sources=%s", failures)
        return 1

    logger.info("run_status=success source_count=%s", len(selected_sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(ingest_data())


