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
# CHANGE ME: Candidate policy presets for weights. Keep visible/easy to hard-code after policy alignment.
WEIGHT_PRESETS = {
    "Balanced (v1)": {
        "backlog_component": 25,
        "aging_component": 25,
        "repeat_component": 25,
        "resolution_component": 25,
    },
    "Resident Experience": {
        "backlog_component": 35,
        "aging_component": 40,
        "repeat_component": 15,
        "resolution_component": 10,
    },
    "Operational Efficiency": {
        "backlog_component": 20,
        "aging_component": 15,
        "repeat_component": 30,
        "resolution_component": 35,
    },
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


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        equal_weight = 1.0 / len(COMPONENT_COLUMNS)
        return {column: equal_weight for column in COMPONENT_COLUMNS}
    return {column: raw_weights.get(column, 0.0) / total for column in COMPONENT_COLUMNS}


def calculate_simulated_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if df.empty:
        return df

    scored = df.copy()
    weighted_sum = sum(scored[column].astype(float) * weights.get(column, 0.0) for column in COMPONENT_COLUMNS)
    scored["simulated_frustration_index"] = weighted_sum.round(2)
    return scored


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


st.set_page_config(page_title="SD Neighborhood Pulse", layout="wide")
st.title("San Diego Neighborhood Pulse")
st.caption("Neighborhood service frustration and hotspot monitoring")

if not DB_PATH.exists():
    st.error("DuckDB file not found. Run the manual refresh workflow first.")
    st.stop()

frustration = load_frustration_data()
hotspots = load_hotspots_data()
service_daily = load_service_daily_data()
frustration = normalize_zipcode_columns(frustration)
hotspots = normalize_zipcode_columns(hotspots)
service_daily = normalize_zipcode_columns(service_daily)

frustration["as_of_date"] = pd.to_datetime(frustration["as_of_date"])
if not service_daily.empty:
    service_daily["metric_date"] = pd.to_datetime(service_daily["metric_date"])

if frustration.empty:
    st.warning("No frustration index data available yet. Run dbt models first.")
    st.stop()

latest_date = frustration["as_of_date"].max()
latest_date_display = pd.to_datetime(latest_date).date().isoformat()
st.sidebar.header("Filters")
window = st.sidebar.selectbox("Window (days)", [30, 90], index=0)
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

st.sidebar.markdown("---")
st.sidebar.subheader("Policy Tuning Lab")
preset_name = st.sidebar.selectbox("Weight preset", options=list(WEIGHT_PRESETS.keys()), index=0)
if "weight_initialized" not in st.session_state:
    for component, value in WEIGHT_PRESETS["Balanced (v1)"].items():
        st.session_state[f"weight_{component}"] = int(value)
    st.session_state["weight_initialized"] = True

if st.sidebar.button("Apply preset", use_container_width=True):
    for component, value in WEIGHT_PRESETS[preset_name].items():
        st.session_state[f"weight_{component}"] = int(value)

with st.sidebar.expander("Tune component weights", expanded=False):
    raw_weights = {}
    for component in COMPONENT_COLUMNS:
        raw_weights[component] = float(
            st.slider(
                COMPONENT_LABELS[component],
                min_value=0,
                max_value=100,
                step=1,
                key=f"weight_{component}",
            )
        )

weights = normalize_weights(raw_weights)
weight_caption = " | ".join(
    f"{COMPONENT_LABELS[column]}: {weights[column] * 100:.1f}%" for column in COMPONENT_COLUMNS
)
st.sidebar.caption(f"Normalized active weights -> {weight_caption}")
ranking_column = "simulated_frustration_index"

current_slice_all = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["as_of_date"] == latest_date)
]
current_slice_all = calculate_simulated_index(current_slice_all, weights).sort_values(ranking_column, ascending=False)
current_slice = current_slice_all[current_slice_all["grain_value"] != "Unknown"]
if grain == "comm_plan_name" and not include_reserve:
    current_slice = current_slice[~current_slice["grain_value"].apply(is_reserve_comm_plan)]

if current_slice_all.empty:
    st.warning(
        f"No rows available for {grain} at {window}-day window on {latest_date_display}. "
        "Try a different window or neighborhood grain."
    )
    st.stop()

current_unknown_rate = unknown_rate(current_slice_all, "grain_value")
if current_unknown_rate > 0:
    st.warning(
        f"Data quality note: {current_unknown_rate:.1f}% of {GRAIN_LABELS[grain]} rows are mapped to Unknown in this current cut."
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
trend = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["grain_value"] == selected_value)
].sort_values("as_of_date")
trend = calculate_simulated_index(trend, weights)

if trend.empty:
    st.info("No trend points available for the selected area.")
    st.stop()

city_slice = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["grain_value"] != "Unknown")
]
if grain == "comm_plan_name" and not include_reserve:
    city_slice = city_slice[~city_slice["grain_value"].apply(is_reserve_comm_plan)]
city_slice = calculate_simulated_index(city_slice, weights)
city_median_latest = float(current_slice[ranking_column].median())
latest_index = float(selected_latest[ranking_column])
latest_requests = float(selected_latest["request_count"])
prior_week_index_avg = prior_week_average(trend, ranking_column, latest_date)
prior_week_request_avg = prior_week_average(trend, "request_count", latest_date)
delta_vs_prior_week = latest_index - prior_week_index_avg if prior_week_index_avg is not None else None
request_delta_vs_prior_week = latest_requests - prior_week_request_avg if prior_week_request_avg is not None else None
delta_vs_city_median = latest_index - city_median_latest
top_driver = driver_label(selected_latest)

focused_tab, global_tab = st.tabs(["Focused Area", "Global Landscape"])

with focused_tab:
    st.subheader("What changed this week")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Selected area", selected_value)
    kpi2.metric(
        "Frustration index",
        f"{latest_index:.2f}",
        delta=(f"{delta_vs_prior_week:+.2f} vs prior 7d avg" if delta_vs_prior_week is not None else None),
    )
    kpi3.metric(
        "Vs city median",
        f"{delta_vs_city_median:+.2f}",
        delta=f"city median {city_median_latest:.2f}",
    )
    kpi4.metric(
        "Requests",
        f"{int(latest_requests)}",
        delta=(f"{request_delta_vs_prior_week:+.1f} vs prior 7d avg" if request_delta_vs_prior_week is not None else None),
    )
    if delta_vs_prior_week is None:
        st.caption("Not enough history yet for a complete prior 7-day baseline.")

    st.subheader("Headline insight")
    city_direction = "above" if delta_vs_city_median >= 0 else "below"
    if delta_vs_prior_week is None:
        weekly_direction_text = "with limited week-over-week history"
    else:
        weekly_direction_text = "increasing week-over-week" if delta_vs_prior_week >= 0 else "decreasing week-over-week"
    st.markdown(
        f"**{selected_value} is {abs(delta_vs_city_median):.2f} points {city_direction} the city median, "
        f"primarily driven by {top_driver}, and is currently {weekly_direction_text}.**"
    )
    st.caption(
        f"Latest index: {latest_index:.2f} | City median: {city_median_latest:.2f} | "
        f"Top component score ({top_driver}): {float(selected_latest[COMPONENT_COLUMNS].max()):.2f}"
    )

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
        st.dataframe(action_table, use_container_width=True, hide_index=True)

    current_start, current_end, baseline_start, baseline_end = baseline_period_bounds(latest_date, window, baseline_mode)
    baseline_components = trend[
        (trend["as_of_date"] >= baseline_start) & (trend["as_of_date"] <= baseline_end)
    ]
    if not baseline_components.empty:
        baseline_component_values = baseline_components[COMPONENT_COLUMNS].mean()
        component_delta = pd.DataFrame(
            {
                "component": COMPONENT_COLUMNS,
                "current_score": [float(selected_latest[column]) for column in COMPONENT_COLUMNS],
                "baseline_score": [float(baseline_component_values[column]) for column in COMPONENT_COLUMNS],
            }
        )
        component_delta["delta"] = component_delta["current_score"] - component_delta["baseline_score"]
        component_delta["weighted_delta"] = component_delta["component"].map(weights) * component_delta["delta"]
        component_delta["component_label"] = component_delta["component"].replace(COMPONENT_LABELS)

        fig_component_delta = px.bar(
            component_delta.sort_values("weighted_delta", ascending=False),
            x="weighted_delta",
            y="component_label",
            orientation="h",
            color="weighted_delta",
            color_continuous_scale="RdYlGn_r",
            title=f"What moved the score vs {BASELINE_LABELS[baseline_mode]}",
            labels={"weighted_delta": "Weighted contribution delta", "component_label": "Component"},
        )
        st.plotly_chart(fig_component_delta, width="stretch")

    st.subheader("Trend")
    show_city_median = st.checkbox("Overlay city median trend", value=True)
    with st.expander("Optional component overlays", expanded=False):
        component_selection = st.multiselect(
            "Select components to overlay",
            options=COMPONENT_COLUMNS,
            default=[],
            format_func=lambda col: COMPONENT_LABELS[col],
        )

    trend_frames = [
        trend[["as_of_date", ranking_column]].rename(columns={ranking_column: "score"}).assign(metric_label="Frustration Index (simulated)")
    ]

    if show_city_median:
        city_median_trend = (
            city_slice.groupby("as_of_date", as_index=False)[ranking_column].median().rename(columns={ranking_column: "score"})
            .assign(metric_label="City Median")
        )
        trend_frames.append(city_median_trend)

    if component_selection:
        component_long = trend[["as_of_date", *component_selection]].melt(
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
        title=f"Frustration trend: {selected_value}",
    )
    fig_trend.update_layout(yaxis_title="Score (0-100)", xaxis_title="As of date", legend_title_text="Series")
    st.plotly_chart(fig_trend, width="stretch")

with global_tab:
    st.caption(
        f"As of {latest_date_display} | Areas in cut: {len(current_slice)} | "
        f"Highest area now: {str(current_slice.iloc[0]['grain_value'])}"
    )

    st.subheader(f"Current Frustration Index ({GRAIN_LABELS[grain]}, {window}-day)")
    if grain == "comm_plan_name":
        reserve_note = "included" if include_reserve else "excluded"
        st.caption(f"Reserve area is {reserve_note} in this community plan view.")
    top_ranked = (
        current_slice.sort_values(ranking_column, ascending=False)
        .head(top_n)
        .sort_values(ranking_column, ascending=True)
    )
    top_ranked = top_ranked.copy()
    top_ranked["grain_value_display"] = top_ranked["grain_value"].astype(str)

    fig_bar = px.bar(
        top_ranked,
        x=ranking_column,
        y="grain_value_display",
        orientation="h",
        color=ranking_column,
        color_continuous_scale="Reds",
        title=f"Top {top_n} areas by simulated frustration index",
        labels={
            ranking_column: "Frustration Index (0-100)",
            "grain_value_display": GRAIN_LABELS[grain],
        },
    )
    chart_height = min(1400, max(560, int(len(top_ranked) * 28)))
    fig_bar.update_layout(height=chart_height, coloraxis_showscale=False)
    fig_bar.update_yaxes(type="category")
    st.plotly_chart(fig_bar, width="stretch")

    with st.expander("How to read the frustration index", expanded=False):
        st.markdown(
            "- The **frustration index** is currently a weighted average of 4 components (0-100 each).\n"
            "- **Higher is worse** across all components.\n"
            "- Components: backlog pressure, aging open cases (>14 days), repeat requests, and resolution lag.\n"
            "- Use **Policy Tuning Lab** to test how different priorities reshape neighborhood rankings."
        )

    baseline_rank = (
        current_slice.sort_values("frustration_index", ascending=False)[["grain_value", "frustration_index"]]
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "baseline_rank"})
    )
    baseline_rank["baseline_rank"] = baseline_rank["baseline_rank"] + 1
    simulated_rank = (
        current_slice.sort_values(ranking_column, ascending=False)[["grain_value", ranking_column]]
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "simulated_rank"})
    )
    simulated_rank["simulated_rank"] = simulated_rank["simulated_rank"] + 1
    rank_delta = baseline_rank.merge(simulated_rank, on="grain_value", how="inner")
    rank_delta["rank_shift"] = rank_delta["baseline_rank"] - rank_delta["simulated_rank"]

    st.subheader("Policy tuning impact")
    movers = rank_delta.reindex(rank_delta["rank_shift"].abs().sort_values(ascending=False).index).head(10)
    movers = movers.rename(
        columns={
            "grain_value": GRAIN_LABELS[grain],
            "baseline_rank": "Baseline Rank (v1)",
            "simulated_rank": "Simulated Rank",
            "rank_shift": "Rank Shift (+ rises)",
        }
    )
    st.dataframe(movers, use_container_width=True, hide_index=True)

    st.subheader("Hotspots")
    hotspot_slice = hotspots[hotspots["window_days"] == window].copy()
    hotspot_slice["grain_value"] = hotspot_slice[grain].fillna("Unknown")
    hotspot_slice = hotspot_slice[hotspot_slice["grain_value"] != "Unknown"]
    geojson, geojson_path = load_boundary_geojson(grain)
    boundary_join_key = None
    if geojson is not None:
        geojson = filter_geojson_for_grain(geojson, grain)
        boundary_join_key = detect_join_key(geojson, set(current_slice["grain_value"].astype(str)), grain)

    if geojson is not None and boundary_join_key is not None:
        choropleth_data = current_slice[["grain_value", "frustration_index"]].copy()
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
            color="frustration_index",
            color_continuous_scale="Reds",
            map_style="carto-positron",
            zoom=9.5,
            center={"lat": 32.7157, "lon": -117.1611},
            opacity=0.7,
            labels={"frustration_index": "Frustration Index"},
            title=f"Choropleth by {GRAIN_LABELS[grain]} (latest {window}-day index)",
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

st.info(
    "Manual refresh: run ingest + dbt run/test/snapshot from README before checking this dashboard. "
    "The app reads from data/db/city_health.duckdb."
)
