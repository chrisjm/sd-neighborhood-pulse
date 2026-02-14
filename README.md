 # SD Neighborhood Pulse

 This project ingests San Diego Get It Done request feeds, models them in dbt + DuckDB, snapshots lifecycle changes, and publishes neighborhood-level hotspot and frustration metrics for reporting.

 ## Data Pipeline Overview
1. **Ingest CSVs** (`scripts/ingest.py`) from:
   - **Daily refresh:** `open_current`, `closed_2026`
   - **Backfill-once (auto-download when missing in `data/raw/latest`):**
     - `closed_2025`
     - `closed_2024`
     - `closed_2023`
     - `closed_2022`
     - `closed_2021`
     - `closed_2020`
     - `closed_2019`
     - `closed_2018`
     - `closed_2017`
 2. **Build dbt models** for source, staging, intermediate, and marts layers.
 3. **Run dbt snapshots** for request state history.
 4. **Visualize** via notebook and Streamlit app.

 ## Phase 3 Outputs
 - Snapshot: `snp_get_it_done_requests`
 - Hotspots: `fct_request_hotspots` (30-day and 90-day windows)
 - Daily metrics: `fct_neighborhood_daily_metrics`
 - Frustration index: `fct_neighborhood_frustration_index`
   - Grains: `comm_plan_name` (primary), `council_district`, `zipcode`
   - Equal-weight components (v1): backlog, aging (>14 days), repeat pressure, resolution lag

 ## Manual Daily Refresh (v1)
Run from project root:

 ```bash
 uv run python scripts/ingest.py --source all
 uv run dbt run --project-dir city_health --profiles-dir .
 uv run dbt test --project-dir city_health --profiles-dir .
 uv run dbt snapshot --project-dir city_health --profiles-dir .
```

## Scheduled Daily Refresh (cron/systemd)
For now, orchestration is intentionally lightweight and dbt-friendly.

### Cron example
Run every day at 5:10 AM:

```bash
10 5 * * * cd /home/chris/code/sd-neighborhood-pulse && uv run python scripts/ingest.py --source all && uv run dbt run --project-dir city_health --profiles-dir . && uv run dbt test --project-dir city_health --profiles-dir . && uv run dbt snapshot --project-dir city_health --profiles-dir .
```

Notes:
- Backfill datasets are skipped once `data/raw/latest/get_it_done_<source>_latest.csv` exists.
- Use `--force` to re-download any source.

 ## Visualization

 ### Notebook
 Open and run:
 - `notebooks/phase3_analysis.ipynb`

 ### Streamlit
 Run from project root:

 ```bash
 uv run streamlit run apps/phase3_dashboard.py
 ```

 The app reads from `data/db/city_health.duckdb` and expects dbt marts to be built.
