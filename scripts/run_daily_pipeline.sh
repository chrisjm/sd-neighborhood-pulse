#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INGEST_ARGS=(--source all)
if [[ "${1:-}" == "--force" ]]; then
  INGEST_ARGS+=(--force)
fi

cd "$PROJECT_ROOT"

uv run python scripts/ingest.py "${INGEST_ARGS[@]}"
uv run dbt run --project-dir city_health --profiles-dir .
uv run dbt test --project-dir city_health --profiles-dir .
uv run dbt snapshot --project-dir city_health --profiles-dir .
