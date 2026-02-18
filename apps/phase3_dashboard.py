from pathlib import Path
import json
import re

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/db/city_health.duckdb")
GRAIN_LABELS = {
    "comm_plan_name": "Community Plan",
    "council_district": "Council District",
    "zipcode": "ZIP Code",
}
BOUNDARY_DIR = Path("data/boundaries")
ALT_BOUNDARY_DIR = Path("data/geojson")
BOUNDARY_FILE_CANDIDATES = {
    "comm_plan_name": [
        "comm_plan_codes.geojson",
        "comm_plan_name.geojson",
        "community_plans.geojson",
        "community_plan.geojson",
    ],
    "council_district": ["council_district.geojson", "council_districts.geojson"],
    "zipcode": ["zipcode.geojson", "zipcodes.geojson", "zcta.geojson"],
}
COMPONENT_COLUMNS = [
    "backlog_component",
    "aging_component",
    "repeat_component",
    "resolution_component",
]
COMPONENT_LABELS = {
    "backlog_component": "Backlog Pressure",
    "aging_component": "Aging Open Cases",
    "repeat_component": "Repeat Requests",
    "resolution_component": "Resolution Lag",
}
BASELINE_LABELS = {
    "prior_30": "Prior 30 Days",
    "prior_90": "Prior 90 Days",
    "yoy": "Year over Year",
}


@st.cache_data(ttl=300)
def load_frustration_data() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            """
            select *
            from main.fct_neighborhood_frustration_index
            """
        ).fetchdf()


@st.cache_data(ttl=300)
def load_service_daily_data() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select
                    requested_date as metric_date,
                    comm_plan_name,
                    council_district,
                    cast(zipcode as varchar) as zipcode,
                    service_name,
                    count(*) as request_count
                from main.int_requests_enriched_time
                group by 1, 2, 3, 4, 5
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_neighborhood_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_neighborhood_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_city_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_city_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_neighborhood_service_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_neighborhood_service_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_city_service_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_city_service_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_hotspots_data() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            """
            select *
            from main.fct_request_hotspots
            order by window_days, request_count desc
            """
        ).fetchdf()


def unknown_rate(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    return float((df[column] == "Unknown").mean() * 100)


def driver_label(row: pd.Series) -> str:
    return COMPONENT_LABELS[row[COMPONENT_COLUMNS].astype(float).idxmax()]


def prior_week_average(trend_df: pd.DataFrame, value_column: str, latest: pd.Timestamp) -> float | None:
    prior_week = trend_df[
        (trend_df["as_of_date"] < latest) & (trend_df["as_of_date"] >= (latest - pd.Timedelta(days=7)))
    ]
    if prior_week.empty:
        return None
    return float(prior_week[value_column].mean())


def baseline_period_bounds(
    latest_date: pd.Timestamp, current_window_days: int, baseline_mode: str
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    current_end = latest_date
    current_start = latest_date - pd.Timedelta(days=current_window_days - 1)

    if baseline_mode == "prior_30":
        baseline_days = 30
        baseline_end = current_start - pd.Timedelta(days=1)
        baseline_start = baseline_end - pd.Timedelta(days=baseline_days - 1)
    elif baseline_mode == "prior_90":
        baseline_days = 90
        baseline_end = current_start - pd.Timedelta(days=1)
        baseline_start = baseline_end - pd.Timedelta(days=baseline_days - 1)
    else:
        baseline_start = current_start - pd.Timedelta(days=365)
        baseline_end = current_end - pd.Timedelta(days=365)

    return current_start, current_end, baseline_start, baseline_end


def grain_column(grain_type: str) -> str:
    return {
        "comm_plan_name": "comm_plan_name",
        "council_district": "council_district",
        "zipcode": "zipcode",
    }[grain_type]


def recommend_policy_action(service_name: str) -> str:
    # CHANGE ME: Policy playbook mapping can be hard-coded once stakeholders finalize interventions.
    service_text = str(service_name).lower()
    if "pothole" in service_text:
        return "Prioritize arterial-first pothole crews and rebalance dispatch zones."
    if "graffiti" in service_text:
        return "Expand rapid-removal routing and hotspot patrol scheduling."
    if "trash" in service_text or "dump" in service_text:
        return "Increase bulky-item pickup capacity and target chronic dumping blocks."
    if "street light" in service_text or "light" in service_text:
        return "Bundle corridor lighting fixes with outage-cluster escalation."
    return "Audit staffing/route bottlenecks and assign a 2-week service recovery sprint."


def build_service_change_table(
    service_daily: pd.DataFrame,
    grain_type: str,
    selected_value: str,
    latest_date: pd.Timestamp,
    current_window_days: int,
    baseline_mode: str,
) -> pd.DataFrame:
    if service_daily.empty:
        return pd.DataFrame()

    selected_column = grain_column(grain_type)
    area = service_daily[service_daily[selected_column].astype(str) == str(selected_value)].copy()
    if area.empty:
        return pd.DataFrame()

    area["metric_date"] = pd.to_datetime(area["metric_date"])
    current_start, current_end, baseline_start, baseline_end = baseline_period_bounds(
        latest_date, current_window_days, baseline_mode
    )

    current_mask = (area["metric_date"] >= current_start) & (area["metric_date"] <= current_end)
    baseline_mask = (area["metric_date"] >= baseline_start) & (area["metric_date"] <= baseline_end)

    current_counts = (
        area.loc[current_mask]
        .groupby("service_name", as_index=False)["request_count"]
        .sum()
        .rename(columns={"request_count": "current_count"})
    )
    baseline_counts = (
        area.loc[baseline_mask]
        .groupby("service_name", as_index=False)["request_count"]
        .sum()
        .rename(columns={"request_count": "baseline_count"})
    )

    if current_counts.empty:
        return pd.DataFrame()

    merged = current_counts.merge(baseline_counts, on="service_name", how="left").fillna({"baseline_count": 0})
    merged["absolute_change"] = merged["current_count"] - merged["baseline_count"]
    merged["pct_change"] = merged.apply(
        lambda row: None
        if row["baseline_count"] == 0
        else ((row["current_count"] - row["baseline_count"]) / row["baseline_count"]) * 100,
        axis=1,
    )
    merged["recommended_action"] = merged["service_name"].apply(recommend_policy_action)
    return merged.sort_values(["absolute_change", "current_count"], ascending=[False, False])


def build_intervention_priority_table(
    current_slice: pd.DataFrame,
    service_daily: pd.DataFrame,
    hotspot_slice: pd.DataFrame,
    grain_type: str,
    latest_date: pd.Timestamp,
    current_window_days: int,
    baseline_mode: str,
    max_rows: int = 12,
) -> pd.DataFrame:
    if current_slice.empty:
        return pd.DataFrame()

    # Keep business-facing table compact and stable for policy review.
    focus_areas = current_slice.sort_values("request_count", ascending=False).head(max_rows)
    hotspot_summary = pd.DataFrame()
    if not hotspot_slice.empty:
        hotspot_summary = (
            hotspot_slice.groupby("grain_value", as_index=False)
            .agg(
                dominant_service=("dominant_service_name", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
                open_ratio_pct=("open_ratio_pct", "mean"),
            )
            .fillna({"dominant_service": "Unknown", "open_ratio_pct": 0.0})
        )

    rows: list[dict[str, object]] = []
    for _, area_row in focus_areas.iterrows():
        area_name = str(area_row["grain_value"])
        service_change = build_service_change_table(
            service_daily=service_daily,
            grain_type=grain_type,
            selected_value=area_name,
            latest_date=latest_date,
            current_window_days=current_window_days,
            baseline_mode=baseline_mode,
        )

        if service_change.empty:
            trend_pct = None
            suggested_action = "Insufficient service-level baseline data."
            service_name = "Unknown"
        else:
            top_service = service_change.iloc[0]
            service_name = str(top_service["service_name"])
            trend_pct = top_service["pct_change"]
            suggested_action = str(top_service["recommended_action"])

        aging_value = area_row.get("aging_component", 0.0)
        if pd.isna(aging_value):
            aging_value = 0.0
        row = {
            "Area": area_name,
            "Requests": round(float(area_row["request_count"]), 0),
            "Aging Burden": round(float(aging_value), 2),
            "Dominant Service": service_name,
            "Trend %": trend_pct,
            "Open Ratio %": None,
            "Suggested Action": suggested_action,
        }
        rows.append(row)

    output = pd.DataFrame(rows)
    if not hotspot_summary.empty:
        output = output.merge(
            hotspot_summary.rename(columns={"grain_value": "Area", "dominant_service": "Hotspot Dominant Service"}),
            on="Area",
            how="left",
        )
        output["Dominant Service"] = output["Dominant Service"].where(
            output["Dominant Service"] != "Unknown", output["Hotspot Dominant Service"]
        )
        output["Open Ratio %"] = output["open_ratio_pct"]
        output = output.drop(columns=["Hotspot Dominant Service", "open_ratio_pct"])

    output["Trend %"] = output["Trend %"].map(
        lambda value: "new" if value is None or pd.isna(value) else f"{float(value):+.1f}%"
    )
    output["Open Ratio %"] = output["Open Ratio %"].map(
        lambda value: "n/a" if value is None or pd.isna(value) else f"{float(value):.1f}%"
    )
    return output


def load_boundary_geojson(grain_type: str) -> tuple[dict | None, str | None]:
    candidates = BOUNDARY_FILE_CANDIDATES.get(grain_type, [])
    for filename in candidates:
        for boundary_dir in (ALT_BOUNDARY_DIR, BOUNDARY_DIR):
            boundary_path = boundary_dir / filename
            if boundary_path.exists():
                with boundary_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh), str(boundary_path)
    return None, None


def normalize_join_value(value: object, grain_type: str) -> str:
    text = str(value).strip()
    if text == "":
        return ""

    if grain_type == "comm_plan_name":
        text = text.lower().replace("&", " and ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text.startswith("reserve area"):
            return "reserve"
        return text

    if grain_type == "council_district":
        match = re.search(r"(\d{1,2})", text)
        if match:
            return str(int(match.group(1)))

    return text


def is_reserve_comm_plan(value: object) -> bool:
    return normalize_join_value(value, "comm_plan_name") == "reserve"


def build_geojson_join_lookup(geojson: dict, join_key: str, grain_type: str) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for feature in geojson.get("features", []):
        raw_value = feature.get("properties", {}).get(join_key)
        if raw_value is None:
            continue
        normalized_value = normalize_join_value(raw_value, grain_type)
        if normalized_value and normalized_value not in lookup:
            lookup[normalized_value] = raw_value
    return lookup


def detect_join_key(geojson: dict, candidate_values: set[str], grain_type: str) -> str | None:
    features = geojson.get("features", [])
    if not features:
        return None

    normalized_candidates = {normalize_join_value(value, grain_type) for value in candidate_values}
    normalized_candidates.discard("")

    sample_props = features[0].get("properties", {})
    best_key = None
    best_overlap = 0
    best_score = 0.0
    for key in sample_props:
        geo_values = {
            normalize_join_value(feature.get("properties", {}).get(key, ""), grain_type)
            for feature in features
            if feature.get("properties", {}).get(key) is not None
        }
        overlap = len(normalized_candidates & geo_values)
        if overlap == 0:
            continue

        denominator = max(len(normalized_candidates), len(geo_values), 1)
        score = overlap / denominator
        if score > best_score or (score == best_score and overlap > best_overlap):
            best_score = score
            best_overlap = overlap
            best_key = key
    return best_key if best_overlap > 0 else None


def geojson_property_is_numeric(geojson: dict, key: str) -> bool:
    for feature in geojson.get("features", []):
        value = feature.get("properties", {}).get(key)
        if value is not None:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def filter_geojson_for_grain(geojson: dict, grain_type: str) -> dict:
    if grain_type != "council_district":
        return geojson

    features = geojson.get("features", [])
    if not features:
        return geojson

    sample_props = features[0].get("properties", {})
    if "JUR_NAME" not in sample_props:
        return geojson

    filtered_features = [
        feature
        for feature in features
        if str(feature.get("properties", {}).get("JUR_NAME", "")).strip().upper() == "SAN DIEGO"
    ]
    if not filtered_features:
        return geojson

    return {**geojson, "features": filtered_features}


def normalize_zipcode_value(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    if text == "" or text.lower() == "unknown":
        return "Unknown"

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text.isdigit() and len(text) <= 5:
        return text.zfill(5)

    return text


def normalize_zipcode_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if "zipcode" in normalized.columns:
        normalized["zipcode"] = normalized["zipcode"].apply(normalize_zipcode_value)

    if {"grain_type", "grain_value"}.issubset(normalized.columns):
        zip_mask = normalized["grain_type"] == "zipcode"
        normalized.loc[zip_mask, "grain_value"] = normalized.loc[zip_mask, "grain_value"].apply(normalize_zipcode_value)

    return normalized


def build_window_request_rollup(
    daily_df: pd.DataFrame, grain_type: str, latest_date: pd.Timestamp | None, window_days: int
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    slice_df = daily_df[daily_df.get("grain_type") == grain_type].copy()
    if slice_df.empty:
        return pd.DataFrame()

    slice_df["metric_date"] = pd.to_datetime(slice_df["metric_date"])
    resolved_latest = latest_date or slice_df["metric_date"].max()
    if pd.isna(resolved_latest):
        return pd.DataFrame()

    window_start = resolved_latest - pd.Timedelta(days=window_days - 1)
    window_mask = (slice_df["metric_date"] >= window_start) & (slice_df["metric_date"] <= resolved_latest)
    rollup = (
        slice_df.loc[window_mask]
        .groupby("grain_value", as_index=False)["request_count"]
        .sum()
        .assign(
            as_of_date=resolved_latest,
            window_days=window_days,
            grain_type=grain_type,
        )
    )
    for column in COMPONENT_COLUMNS:
        rollup[column] = pd.NA
    return rollup


def window_unknown_rate(
    daily_df: pd.DataFrame,
    grain_type: str,
    latest_date: pd.Timestamp | None,
    window_days: int,
) -> float | None:
    if daily_df.empty:
        return None

    slice_df = daily_df[daily_df.get("grain_type") == grain_type].copy()
    if slice_df.empty:
        return None

    slice_df["metric_date"] = pd.to_datetime(slice_df["metric_date"])
    resolved_latest = latest_date or slice_df["metric_date"].max()
    if pd.isna(resolved_latest):
        return None

    window_start = resolved_latest - pd.Timedelta(days=window_days - 1)
    window_mask = (slice_df["metric_date"] >= window_start) & (slice_df["metric_date"] <= resolved_latest)
    window_slice = slice_df.loc[window_mask]
    if window_slice.empty:
        return None

    total = window_slice["request_count"].sum()
    if total == 0:
        return 0.0
    unknown_total = window_slice.loc[window_slice["grain_value"] == "Unknown", "request_count"].sum()
    return float((unknown_total / total) * 100)


def window_total_for_area(
    daily_df: pd.DataFrame,
    grain_type: str,
    grain_value: str,
    latest_date: pd.Timestamp | None,
    window_days: int,
    offset_days: int = 0,
) -> float | None:
    if daily_df.empty:
        return None

    slice_df = daily_df[
        (daily_df.get("grain_type") == grain_type)
        & (daily_df.get("grain_value").astype(str) == str(grain_value))
    ].copy()
    if slice_df.empty:
        return None

    slice_df["metric_date"] = pd.to_datetime(slice_df["metric_date"])
    resolved_latest = latest_date or slice_df["metric_date"].max()
    if pd.isna(resolved_latest):
        return None

    window_end = resolved_latest - pd.Timedelta(days=offset_days)
    window_start = window_end - pd.Timedelta(days=window_days - 1)
    window_mask = (slice_df["metric_date"] >= window_start) & (slice_df["metric_date"] <= window_end)
    total = slice_df.loc[window_mask, "request_count"].sum()
    return float(total)


def summarize_busy_light_services(service_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if service_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    busy = service_df[service_df["is_busy_day"]].sort_values("request_count", ascending=False)
    light = service_df[service_df["is_light_day"]].sort_values("request_count", ascending=False)
    return busy, light


def resolve_latest_metric_date(preferred: pd.Timestamp, df: pd.DataFrame) -> pd.Timestamp | None:
    if df.empty or "metric_date" not in df.columns:
        return None

    metric_dates = pd.to_datetime(df["metric_date"]).dropna()
    if metric_dates.empty:
        return None

    if preferred in metric_dates.values:
        return preferred

    return metric_dates.max()


def resolve_latest_slice_date(
    df: pd.DataFrame,
    grain_type: str,
    grain_value: str,
    preferred: pd.Timestamp,
) -> pd.Timestamp | None:
    if df.empty:
        return None

    slice_df = df[
        (df.get("grain_type") == grain_type)
        & (df.get("grain_value").astype(str) == str(grain_value))
    ]
    if slice_df.empty:
        return None

    return resolve_latest_metric_date(preferred, slice_df)


def read_metric_value(row_df: pd.DataFrame, column: str) -> float | None:
    if row_df.empty or column not in row_df.columns:
        return None

    value = row_df.iloc[0][column]
    if pd.isna(value):
        return None

    return float(value)


st.set_page_config(page_title="SD Neighborhood Pulse", layout="wide")
st.title("San Diego Neighborhood Pulse")
st.caption("Weekly briefing on neighborhood service pressure and demand patterns")

if not DB_PATH.exists():
    st.error("DuckDB file not found. Run the manual refresh workflow first.")
    st.stop()

frustration = load_frustration_data()
hotspots = load_hotspots_data()
service_daily = load_service_daily_data()
neighborhood_daily_metrics = load_neighborhood_daily_metrics()
city_daily_metrics = load_city_daily_metrics()
neighborhood_service_daily = load_neighborhood_service_daily_metrics()
city_service_daily = load_city_service_daily_metrics()
frustration = normalize_zipcode_columns(frustration)
hotspots = normalize_zipcode_columns(hotspots)
service_daily = normalize_zipcode_columns(service_daily)
neighborhood_daily_metrics = normalize_zipcode_columns(neighborhood_daily_metrics)
neighborhood_service_daily = normalize_zipcode_columns(neighborhood_service_daily)

frustration["as_of_date"] = pd.to_datetime(frustration["as_of_date"])
if not service_daily.empty:
    service_daily["metric_date"] = pd.to_datetime(service_daily["metric_date"])
if not neighborhood_daily_metrics.empty:
    neighborhood_daily_metrics["metric_date"] = pd.to_datetime(neighborhood_daily_metrics["metric_date"])
if not city_daily_metrics.empty:
    city_daily_metrics["metric_date"] = pd.to_datetime(city_daily_metrics["metric_date"])
if not neighborhood_service_daily.empty:
    neighborhood_service_daily["metric_date"] = pd.to_datetime(neighborhood_service_daily["metric_date"])
if not city_service_daily.empty:
    city_service_daily["metric_date"] = pd.to_datetime(city_service_daily["metric_date"])

if frustration.empty:
    st.warning("No neighborhood pressure data available yet. Run dbt models first.")
    st.stop()

latest_date = frustration["as_of_date"].max()
latest_date_display = pd.to_datetime(latest_date).date().isoformat()
st.sidebar.header("Filters")
window = st.sidebar.selectbox("Window (days)", [7, 30, 90], index=0)
grain = st.sidebar.selectbox("Neighborhood grain", ["comm_plan_name", "council_district", "zipcode"], index=0)
baseline_mode = st.sidebar.selectbox(
    "Comparison baseline",
    options=list(BASELINE_LABELS.keys()),
    format_func=lambda value: BASELINE_LABELS[value],
    index=0,
)
st.sidebar.caption("Showing all areas for the selected grain and window.")
include_reserve = False
if grain == "comm_plan_name":
    include_reserve = st.sidebar.checkbox("Include Reserve (power users)", value=False)
    if include_reserve:
        st.sidebar.caption("Including Reserve area in community plan views.")
    else:
        st.sidebar.caption("Reserve is excluded by default. Enable the checkbox to include it.")

component_window = 30 if window == 7 else window
latest_daily_date = resolve_latest_metric_date(latest_date, neighborhood_daily_metrics)
if window == 7:
    current_slice_all = build_window_request_rollup(neighborhood_daily_metrics, grain, latest_daily_date, window)
else:
    current_slice_all = frustration[
        (frustration["window_days"] == window)
        & (frustration["grain_type"] == grain)
        & (frustration["as_of_date"] == latest_date)
    ]
current_slice_all = current_slice_all.sort_values("request_count", ascending=False)
current_slice = current_slice_all[current_slice_all["grain_value"] != "Unknown"]
if grain == "comm_plan_name" and not include_reserve:
    current_slice = current_slice[~current_slice["grain_value"].apply(is_reserve_comm_plan)]

if current_slice_all.empty:
    st.warning(
        f"No rows available for {grain} at {window}-day window on {latest_date_display}. "
        "Try a different window or neighborhood grain."
    )
    st.stop()

if window == 7:
    current_unknown_rate = window_unknown_rate(neighborhood_daily_metrics, grain, latest_daily_date, window)
else:
    current_unknown_rate = unknown_rate(current_slice_all, "grain_value")
if current_unknown_rate is not None and current_unknown_rate > 0:
    st.warning(
        f"Data quality note: {current_unknown_rate:.1f}% of {GRAIN_LABELS[grain]} requests are mapped to Unknown in this cut."
    )

if current_slice.empty:
    if grain == "comm_plan_name" and not include_reserve:
        st.info("No non-Reserve community plan rows available for this cut. Enable 'Include Reserve' to include it.")
    else:
        st.info("All rows are currently Unknown after cleaning for this cut; switch grain/window to continue exploration.")
    st.stop()

non_zero_grain_count = int((current_slice["request_count"] > 0).sum())
default_top_n = non_zero_grain_count if non_zero_grain_count > 0 else int(len(current_slice))
top_n = st.sidebar.slider(
    "Top N neighborhoods",
    min_value=1,
    max_value=int(len(current_slice)),
    value=default_top_n,
    step=1,
)

selected_options = current_slice["grain_value"].tolist()
selected_value = st.sidebar.selectbox(
    f"Focus {GRAIN_LABELS[grain]}",
    options=selected_options,
    index=0,
)

selected_latest = current_slice[current_slice["grain_value"] == selected_value].iloc[0]
component_slice_latest = frustration[
    (frustration["window_days"] == component_window)
    & (frustration["grain_type"] == grain)
    & (frustration["as_of_date"] == latest_date)
]
component_slice_latest = component_slice_latest[component_slice_latest["grain_value"] != "Unknown"]
if grain == "comm_plan_name" and not include_reserve:
    component_slice_latest = component_slice_latest[
        ~component_slice_latest["grain_value"].apply(is_reserve_comm_plan)
    ]
selected_component_latest = component_slice_latest[
    component_slice_latest["grain_value"].astype(str) == str(selected_value)
]

component_trend = frustration[
    (frustration["window_days"] == component_window)
    & (frustration["grain_type"] == grain)
    & (frustration["grain_value"] == selected_value)
].sort_values("as_of_date")

if neighborhood_daily_metrics.empty:
    focus_daily_slice = pd.DataFrame()
else:
    focus_daily_slice = neighborhood_daily_metrics[
        (neighborhood_daily_metrics.get("grain_type") == grain)
        & (neighborhood_daily_metrics.get("grain_value").astype(str) == str(selected_value))
    ].copy()
focus_daily_date = resolve_latest_slice_date(neighborhood_daily_metrics, grain, selected_value, latest_date)
if focus_daily_date is not None:
    focus_daily_row = focus_daily_slice[focus_daily_slice["metric_date"] == focus_daily_date]
else:
    focus_daily_row = pd.DataFrame()

city_daily_date = resolve_latest_metric_date(latest_date, city_daily_metrics)
if city_daily_date is not None:
    city_daily_row = city_daily_metrics[city_daily_metrics["metric_date"] == city_daily_date]
else:
    city_daily_row = pd.DataFrame()

if neighborhood_service_daily.empty:
    focus_service_slice = pd.DataFrame()
else:
    focus_service_slice = neighborhood_service_daily[
        (neighborhood_service_daily.get("grain_type") == grain)
        & (neighborhood_service_daily.get("grain_value").astype(str) == str(selected_value))
    ].copy()
focus_service_date = resolve_latest_metric_date(latest_date, focus_service_slice)
if focus_service_date is not None:
    focus_service_latest = focus_service_slice[focus_service_slice["metric_date"] == focus_service_date]
else:
    focus_service_latest = pd.DataFrame()

city_service_date = resolve_latest_metric_date(latest_date, city_service_daily)
if city_service_date is not None:
    city_service_latest = city_service_daily[city_service_daily["metric_date"] == city_service_date]
else:
    city_service_latest = pd.DataFrame()

if component_trend.empty:
    st.info("No component trend points available for the selected area.")
    st.stop()

latest_requests = float(selected_latest["request_count"])
if window == 7:
    prior_window_request_avg = window_total_for_area(
        neighborhood_daily_metrics,
        grain,
        selected_value,
        latest_daily_date,
        window_days=7,
        offset_days=7,
    )
else:
    prior_window_request_avg = prior_week_average(component_trend, "request_count", latest_date)
request_delta_vs_prior_window = (
    latest_requests - prior_window_request_avg if prior_window_request_avg is not None else None
)

if selected_component_latest.empty:
    selected_component_latest = pd.DataFrame([{}])
top_driver = (
    driver_label(selected_component_latest.iloc[0])
    if not selected_component_latest.empty and not selected_component_latest.isna().all(axis=None)
    else "Unavailable"
)

st.subheader("Citywide context")
citywide_daily = neighborhood_daily_metrics[
    (neighborhood_daily_metrics.get("grain_type") == grain)
    & (neighborhood_daily_metrics.get("grain_value") != "Unknown")
].copy()
if not citywide_daily.empty:
    citywide_daily = citywide_daily.sort_values("metric_date")
    citywide_summary = (
        citywide_daily.groupby("metric_date", as_index=False)["request_count"].sum().rename(columns={"request_count": "requests"})
    )
    citywide_summary = citywide_summary.tail(365)
    fig_city = px.line(
        citywide_summary,
        x="metric_date",
        y="requests",
        title=f"Citywide request volume trend ({GRAIN_LABELS[grain]})",
        labels={"metric_date": "Date", "requests": "Total requests"},
    )
    fig_city.update_layout(yaxis_title="Requests")
    st.plotly_chart(fig_city, width="stretch")
else:
    st.info("Citywide trend unavailable for this grain.")

focused_tab, global_tab = st.tabs(["Focused Area", "Global Landscape"])

with focused_tab:
    st.subheader("What changed this week")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Selected area", selected_value)
    repeat_value = read_metric_value(selected_component_latest, "repeat_component")
    resolution_value = read_metric_value(selected_component_latest, "resolution_component")
    kpi2.metric(
        f"Repeat pressure ({component_window}d)",
        f"{repeat_value:.2f}" if repeat_value is not None else "n/a",
    )
    kpi3.metric(
        f"Resolution lag ({component_window}d)",
        f"{resolution_value:.2f}" if resolution_value is not None else "n/a",
    )
    kpi4.metric(
        "Requests",
        f"{int(latest_requests)}",
        delta=(
            f"{request_delta_vs_prior_window:+.1f} vs prior window" if request_delta_vs_prior_window is not None else None
        ),
    )
    recency_focus_open_3d = read_metric_value(focus_daily_row, "opened_request_count_3d")
    recency_focus_open_7d = read_metric_value(focus_daily_row, "opened_request_count_7d")
    recency_focus_closed_3d = read_metric_value(focus_daily_row, "closed_request_count_3d")
    recency_focus_closed_7d = read_metric_value(focus_daily_row, "closed_request_count_7d")
    recency_city_open_3d = read_metric_value(city_daily_row, "opened_request_count_3d")
    recency_city_open_7d = read_metric_value(city_daily_row, "opened_request_count_7d")
    recency_city_closed_3d = read_metric_value(city_daily_row, "closed_request_count_3d")
    recency_city_closed_7d = read_metric_value(city_daily_row, "closed_request_count_7d")
    recency_cols = st.columns(4)
    recency_cols[0].metric(
        "Opened (3d)",
        f"{int(recency_focus_open_3d)}" if recency_focus_open_3d is not None else "n/a",
        delta=(
            f"city {int(recency_city_open_3d)}"
            if recency_city_open_3d is not None and recency_focus_open_3d is not None
            else None
        ),
    )
    recency_cols[1].metric(
        "Opened (7d)",
        f"{int(recency_focus_open_7d)}" if recency_focus_open_7d is not None else "n/a",
        delta=(
            f"city {int(recency_city_open_7d)}"
            if recency_city_open_7d is not None and recency_focus_open_7d is not None
            else None
        ),
    )
    recency_cols[2].metric(
        "Closed (3d)",
        f"{int(recency_focus_closed_3d)}" if recency_focus_closed_3d is not None else "n/a",
        delta=(
            f"city {int(recency_city_closed_3d)}"
            if recency_city_closed_3d is not None and recency_focus_closed_3d is not None
            else None
        ),
    )
    recency_cols[3].metric(
        "Closed (7d)",
        f"{int(recency_focus_closed_7d)}" if recency_focus_closed_7d is not None else "n/a",
        delta=(
            f"city {int(recency_city_closed_7d)}"
            if recency_city_closed_7d is not None and recency_focus_closed_7d is not None
            else None
        ),
    )
    if request_delta_vs_prior_window is None:
        st.caption("Not enough history yet for a complete prior window baseline.")

    st.subheader("Headline insight")
    if request_delta_vs_prior_window is None:
        weekly_direction_text = "with limited week-over-week history"
    else:
        weekly_direction_text = "increasing week-over-week" if request_delta_vs_prior_window >= 0 else "decreasing week-over-week"
    st.markdown(
        f"**{selected_value} recorded {int(latest_requests)} requests in the last {window} days, "
        f"primarily driven by {top_driver}, and is currently {weekly_direction_text}.**"
    )
    if top_driver != "Unavailable":
        top_component_score = (
            selected_component_latest[COMPONENT_COLUMNS]
            .astype(float)
            .max(axis=1)
            .iloc[0]
        )
        st.caption(f"Top component driver ({top_driver}): {top_component_score:.2f}")

    st.subheader("Action brief: what to fix next")
    service_change = build_service_change_table(
        service_daily=service_daily,
        grain_type=grain,
        selected_value=selected_value,
        latest_date=latest_date,
        current_window_days=window,
        baseline_mode=baseline_mode,
    )

    if service_change.empty:
        st.info("Service-level trend breakdown unavailable for this selection yet.")
    else:
        top_service = service_change.iloc[0]
        pct_change_value = top_service["pct_change"]
        if pct_change_value is None or pd.isna(pct_change_value):
            pct_change_text = "new surge vs baseline"
        else:
            pct_change_text = f"{pct_change_value:+.1f}%"

        st.markdown(
            f"**Priority signal:** {top_service['service_name']} requests are {pct_change_text} "
            f"(Δ {int(top_service['absolute_change'])}) using {BASELINE_LABELS[baseline_mode]} as baseline."
        )
        st.caption(f"Suggested intervention: {top_service['recommended_action']}")

        action_table = service_change.head(8).copy()
        action_table["pct_change"] = action_table["pct_change"].map(
            lambda value: "new" if value is None or pd.isna(value) else f"{value:+.1f}%"
        )
        action_table = action_table.rename(
            columns={
                "service_name": "Service",
                "current_count": "Current Volume",
                "baseline_count": f"Baseline Volume ({BASELINE_LABELS[baseline_mode]})",
                "absolute_change": "Absolute Change",
                "pct_change": "% Change",
                "recommended_action": "Recommended Action",
            }
        )
        st.dataframe(action_table, width="stretch", hide_index=True)

    current_start, current_end, baseline_start, baseline_end = baseline_period_bounds(latest_date, window, baseline_mode)
    baseline_components = component_trend[
        (component_trend["as_of_date"] >= baseline_start) & (component_trend["as_of_date"] <= baseline_end)
    ]
    if not baseline_components.empty:
        baseline_component_values = baseline_components[COMPONENT_COLUMNS].mean()
        component_delta = pd.DataFrame(
            {
                "component": COMPONENT_COLUMNS,
                "current_score": [float(selected_component_latest.iloc[0].get(column, 0.0)) for column in COMPONENT_COLUMNS],
                "baseline_score": [float(baseline_component_values[column]) for column in COMPONENT_COLUMNS],
            }
        )
        component_delta["delta"] = component_delta["current_score"] - component_delta["baseline_score"]
        component_delta["component_label"] = component_delta["component"].replace(COMPONENT_LABELS)

        fig_component_delta = px.bar(
            component_delta.sort_values("delta", ascending=False),
            x="delta",
            y="component_label",
            orientation="h",
            color="delta",
            color_continuous_scale="RdYlGn_r",
            title=f"Component shifts vs {BASELINE_LABELS[baseline_mode]}",
            labels={"delta": "Component delta", "component_label": "Component"},
        )
        st.plotly_chart(fig_component_delta, width="stretch")

    st.subheader("Trend")
    with st.expander("Optional component overlays", expanded=False):
        component_selection = st.multiselect(
            "Select components to overlay",
            options=COMPONENT_COLUMNS,
            default=[],
            format_func=lambda col: COMPONENT_LABELS[col],
        )

    trend_frames = [
        component_trend[["as_of_date", "request_count"]]
        .rename(columns={"request_count": "score"})
        .assign(metric_label=f"Requests ({component_window}d)")
    ]

    if component_selection:
        component_long = component_trend[["as_of_date", *component_selection]].melt(
            id_vars=["as_of_date"],
            value_vars=component_selection,
            var_name="metric",
            value_name="score",
        )
        component_long["metric_label"] = component_long["metric"].replace(COMPONENT_LABELS)
        trend_frames.append(component_long[["as_of_date", "score", "metric_label"]])

    trend_long = pd.concat(trend_frames, ignore_index=True)
    fig_trend = px.line(
        trend_long,
        x="as_of_date",
        y="score",
        color="metric_label",
        title=f"Request + component trend: {selected_value}",
    )
    fig_trend.update_layout(yaxis_title="Score / Requests", xaxis_title="As of date", legend_title_text="Series")
    st.plotly_chart(fig_trend, width="stretch")

    st.subheader("Operational pulse (busy/light services)")
    focus_busy, focus_light = summarize_busy_light_services(focus_service_latest)
    if focus_service_latest.empty:
        st.info("Busy/light service signals unavailable for this selection yet.")
    else:
        pulse_cols = st.columns(2)
        with pulse_cols[0]:
            st.caption("Busy services today (top 8)")
            if focus_busy.empty:
                st.write("No busy services flagged.")
            else:
                st.dataframe(
                    focus_busy.head(8)[["service_name", "request_count"]]
                    .rename(columns={"service_name": "Service", "request_count": "Requests"}),
                    width="stretch",
                    hide_index=True,
                )
        with pulse_cols[1]:
            st.caption("Light services today (top 8)")
            if focus_light.empty:
                st.write("No light services flagged.")
            else:
                st.dataframe(
                    focus_light.head(8)[["service_name", "request_count"]]
                    .rename(columns={"service_name": "Service", "request_count": "Requests"}),
                    width="stretch",
                    hide_index=True,
                )

with global_tab:
    st.caption(
        f"As of {latest_date_display} | Areas in cut: {len(current_slice)} | "
        f"Highest area now: {str(current_slice.iloc[0]['grain_value'])}"
    )

    st.subheader(f"Current Request Volume ({GRAIN_LABELS[grain]}, {window}-day)")
    if grain == "comm_plan_name":
        reserve_note = "included" if include_reserve else "excluded"
        st.caption(f"Reserve area is {reserve_note} in this community plan view.")
    top_ranked = (
        current_slice.sort_values("request_count", ascending=False)
        .head(top_n)
        .sort_values("request_count", ascending=True)
    )
    top_ranked = top_ranked.copy()
    top_ranked["grain_value_display"] = top_ranked["grain_value"].astype(str)

    fig_bar = px.bar(
        top_ranked,
        x="request_count",
        y="grain_value_display",
        orientation="h",
        color="request_count",
        color_continuous_scale="Reds",
        title=f"Top {top_n} areas by request volume",
        labels={
            "request_count": "Requests",
            "grain_value_display": GRAIN_LABELS[grain],
        },
    )
    chart_height = min(1400, max(560, int(len(top_ranked) * 28)))
    fig_bar.update_layout(height=chart_height, coloraxis_showscale=False)
    fig_bar.update_yaxes(type="category")
    st.plotly_chart(fig_bar, width="stretch")

    with st.expander("How to read the components", expanded=False):
        st.markdown(
            "- **Request volume** highlights where demand is highest for the selected window.\n"
            "- Components (backlog, aging, repeat, resolution) describe *why* pressure is elevated.\n"
            "- Use the focused tab to see how component shifts relate to recent demand swings."
        )

    st.subheader("Where to intervene first")
    hotspot_slice = hotspots[hotspots["window_days"] == component_window].copy()
    hotspot_slice["grain_value"] = hotspot_slice[grain].fillna("Unknown")
    hotspot_slice = hotspot_slice[hotspot_slice["grain_value"] != "Unknown"]
    intervention_table = build_intervention_priority_table(
        current_slice=current_slice,
        service_daily=service_daily,
        hotspot_slice=hotspot_slice,
        grain_type=grain,
        latest_date=latest_date,
        current_window_days=window,
        baseline_mode=baseline_mode,
        max_rows=min(top_n, 12),
    )
    if intervention_table.empty:
        st.info("Intervention table unavailable for this cut.")
    else:
        st.dataframe(intervention_table, width="stretch", hide_index=True)

    st.subheader("Hotspots")
    geojson, geojson_path = load_boundary_geojson(grain)
    boundary_join_key = None
    if geojson is not None:
        geojson = filter_geojson_for_grain(geojson, grain)
        boundary_join_key = detect_join_key(geojson, set(current_slice["grain_value"].astype(str)), grain)

    if geojson is not None and boundary_join_key is not None:
        choropleth_data = current_slice[["grain_value", "request_count"]].copy()
        if "grain_geo_value" in current_slice.columns:
            choropleth_data["grain_geo_value"] = current_slice["grain_geo_value"].values
        location_column = "grain_value"

        if grain == "comm_plan_name":
            if "grain_geo_value" in choropleth_data.columns:
                choropleth_data["join_location"] = choropleth_data["grain_geo_value"].fillna(choropleth_data["grain_value"])
            else:
                comm_plan_lookup = build_geojson_join_lookup(geojson, boundary_join_key, grain)
                choropleth_data["join_location"] = choropleth_data["grain_value"].apply(
                    lambda value: comm_plan_lookup.get(normalize_join_value(value, grain))
                )
            choropleth_data = choropleth_data[choropleth_data["join_location"].notna()].copy()
            location_column = "join_location"

        if grain == "council_district":
            district_digits = choropleth_data["grain_value"].astype(str).str.extract(r"(\d{1,2})")[0]
            if geojson_property_is_numeric(geojson, boundary_join_key):
                choropleth_data["join_location"] = pd.to_numeric(district_digits, errors="coerce")
            else:
                choropleth_data["join_location"] = district_digits
            choropleth_data = choropleth_data[choropleth_data["join_location"].notna()].copy()
            location_column = "join_location"

        fig_choropleth = px.choropleth_map(
            choropleth_data,
            geojson=geojson,
            locations=location_column,
            featureidkey=f"properties.{boundary_join_key}",
            color="request_count",
            color_continuous_scale="Reds",
            map_style="carto-positron",
            zoom=9.5,
            center={"lat": 32.7157, "lon": -117.1611},
            opacity=0.7,
            labels={"request_count": "Requests"},
            title=f"Choropleth by {GRAIN_LABELS[grain]} (latest {window}-day requests)",
        )
        fig_choropleth.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(fig_choropleth, width="stretch")
        st.caption(f"Using boundaries: {geojson_path} (join key: {boundary_join_key})")
    else:
        if grain == "comm_plan_name":
            st.info("No community-plan GeoJSON configured yet. Showing hotspot centroid map for now.")
        elif geojson is None:
            st.info(
                f"Choropleth unavailable for {GRAIN_LABELS[grain]}: no boundary GeoJSON found in {ALT_BOUNDARY_DIR} or {BOUNDARY_DIR}. "
                "Showing centroid bubble map instead."
            )
        else:
            st.info(
                f"Choropleth unavailable for {GRAIN_LABELS[grain]}: boundary file found but no matching join key. "
                "Showing centroid bubble map instead."
            )

        map_points = hotspot_slice.sort_values("request_count", ascending=False).head(800)
        if map_points.empty:
            st.warning("No hotspot points available for this filter.")
        else:
            fig_points = px.scatter_map(
                map_points,
                lat="centroid_latitude",
                lon="centroid_longitude",
                size="request_count",
                color="open_ratio_pct",
                color_continuous_scale="OrRd",
                hover_name="grain_value",
                hover_data={
                    "request_count": True,
                    "open_ratio_pct": ":.2f",
                    "dominant_service_name": True,
                    "comm_plan_name": True,
                    "council_district": True,
                    "zipcode": True,
                    "centroid_latitude": False,
                    "centroid_longitude": False,
                    "grain_value": False,
                },
                zoom=10,
                center={"lat": 32.7157, "lon": -117.1611},
                map_style="open-street-map",
                title=f"Hotspot centroids by {GRAIN_LABELS[grain]} ({window}-day)",
            )
            fig_points.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
            st.plotly_chart(fig_points, width="stretch")

    st.subheader("Citywide operational pulse")
    city_busy, city_light = summarize_busy_light_services(city_service_latest)
    if city_service_latest.empty:
        st.info("Citywide busy/light service signals unavailable yet.")
    else:
        pulse_cols = st.columns(2)
        with pulse_cols[0]:
            st.caption("Busy services citywide (top 10)")
            if city_busy.empty:
                st.write("No busy services flagged citywide.")
            else:
                st.dataframe(
                    city_busy.head(10)[["service_name", "request_count"]]
                    .rename(columns={"service_name": "Service", "request_count": "Requests"}),
                    width="stretch",
                    hide_index=True,
                )
        with pulse_cols[1]:
            st.caption("Light services citywide (top 10)")
            if city_light.empty:
                st.write("No light services flagged citywide.")
            else:
                st.dataframe(
                    city_light.head(10)[["service_name", "request_count"]]
                    .rename(columns={"service_name": "Service", "request_count": "Requests"}),
                    width="stretch",
                    hide_index=True,
                )

st.info(
    "Manual refresh: run ingest + dbt run/test/snapshot from README before checking this dashboard. "
    "The app reads from data/db/city_health.duckdb."
)
