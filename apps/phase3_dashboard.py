from pathlib import Path
import json

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
    "comm_plan_name": ["comm_plan_name.geojson", "community_plans.geojson", "community_plan.geojson"],
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


def load_boundary_geojson(grain_type: str) -> tuple[dict | None, str | None]:
    candidates = BOUNDARY_FILE_CANDIDATES.get(grain_type, [])
    for filename in candidates:
        for boundary_dir in (ALT_BOUNDARY_DIR, BOUNDARY_DIR):
            boundary_path = boundary_dir / filename
            if boundary_path.exists():
                with boundary_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh), str(boundary_path)
    return None, None


def detect_join_key(geojson: dict, candidate_values: set[str]) -> str | None:
    features = geojson.get("features", [])
    if not features:
        return None

    sample_props = features[0].get("properties", {})
    best_key = None
    best_overlap = 0
    for key in sample_props:
        geo_values = {
            str(feature.get("properties", {}).get(key, "")).strip()
            for feature in features
            if feature.get("properties", {}).get(key) is not None
        }
        overlap = len(candidate_values & geo_values)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
    return best_key if best_overlap > 0 else None


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

PRESETS = {
    "Top 10 Comm Plans (30d)": {"window": 30, "grain": "comm_plan_name", "top_n": 10},
    "Top 10 Council Districts (30d)": {"window": 30, "grain": "council_district", "top_n": 10},
    "Top 10 ZIP Codes (30d)": {"window": 30, "grain": "zipcode", "top_n": 10},
    "Top 20 Comm Plans (90d)": {"window": 90, "grain": "comm_plan_name", "top_n": 20},
    "Custom": None,
}

if not DB_PATH.exists():
    st.error("DuckDB file not found. Run the manual refresh workflow first.")
    st.stop()

frustration = load_frustration_data()
hotspots = load_hotspots_data()
frustration = normalize_zipcode_columns(frustration)
hotspots = normalize_zipcode_columns(hotspots)

if frustration.empty:
    st.warning("No frustration index data available yet. Run dbt models first.")
    st.stop()

latest_date = frustration["as_of_date"].max()
st.sidebar.header("Filters")
preset_name = st.sidebar.selectbox("Quick view", list(PRESETS.keys()), index=0)

if PRESETS[preset_name] is not None:
    preset = PRESETS[preset_name]
    window = preset["window"]
    grain = preset["grain"]
    top_n = preset["top_n"]
    st.sidebar.caption(f"Preset applied: {preset_name}")
else:
    window = st.sidebar.selectbox("Window (days)", [30, 90], index=0)
    grain = st.sidebar.selectbox("Neighborhood grain", ["comm_plan_name", "council_district", "zipcode"], index=0)
    top_n = st.sidebar.slider("Top N neighborhoods", min_value=10, max_value=50, value=20, step=5)

if PRESETS[preset_name] is not None:
    st.sidebar.write("Adjust manually by switching to **Custom**.")

current_slice_all = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["as_of_date"] == latest_date)
].sort_values("frustration_index", ascending=False)
current_slice = current_slice_all[current_slice_all["grain_value"] != "Unknown"]

if current_slice_all.empty:
    st.warning(
        f"No rows available for {grain} at {window}-day window on {latest_date}. "
        "Try a different preset or switch to Custom mode."
    )
    st.stop()

current_unknown_rate = unknown_rate(current_slice_all, "grain_value")
if current_unknown_rate > 0:
    st.warning(
        f"Data quality note: {current_unknown_rate:.1f}% of {GRAIN_LABELS[grain]} rows are mapped to Unknown in this current cut."
    )

if current_slice.empty:
    st.info("All rows are currently Unknown after cleaning for this cut; switch grain/window to continue exploration.")
    st.stop()

selected_options = current_slice["grain_value"].tolist()[:100]
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

if trend.empty:
    st.info("No trend points available for the selected area.")
    st.stop()

city_slice = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["grain_value"] != "Unknown")
]
city_median_latest = float(current_slice["frustration_index"].median())
latest_index = float(selected_latest["frustration_index"])
latest_requests = float(selected_latest["request_count"])
prior_week_index_avg = prior_week_average(trend, "frustration_index", latest_date)
prior_week_request_avg = prior_week_average(trend, "request_count", latest_date)
delta_vs_prior_week = latest_index - prior_week_index_avg if prior_week_index_avg is not None else None
request_delta_vs_prior_week = latest_requests - prior_week_request_avg if prior_week_request_avg is not None else None
delta_vs_city_median = latest_index - city_median_latest
top_driver = driver_label(selected_latest)

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

st.caption(
    f"As of {latest_date} | Areas in cut: {len(current_slice)} | "
    f"Highest area now: {str(current_slice.iloc[0]['grain_value'])}"
)

st.subheader(f"Current Frustration Index ({GRAIN_LABELS[grain]}, {window}-day)")
if PRESETS[preset_name] is None:
    st.caption("Custom mode enabled")
else:
    st.caption(f"Preset mode: {preset_name}")
top_ranked = current_slice.head(top_n).sort_values("frustration_index", ascending=True)
top_ranked = top_ranked.copy()
top_ranked["grain_value_display"] = top_ranked["grain_value"].astype(str)

fig_bar = px.bar(
    top_ranked,
    x="frustration_index",
    y="grain_value_display",
    orientation="h",
    color="frustration_index",
    color_continuous_scale="Reds",
    title=f"Top {top_n} areas by frustration index",
    labels={
        "frustration_index": "Frustration Index (0-100)",
        "grain_value_display": GRAIN_LABELS[grain],
    },
)
fig_bar.update_layout(height=560, coloraxis_showscale=False)
fig_bar.update_yaxes(type="category")
st.plotly_chart(fig_bar, width="stretch")

with st.expander("How to read the frustration index", expanded=False):
    st.markdown(
        "- The **frustration index** is an equal-weight average of 4 components (0-100 each).\n"
        "- **Higher is worse** across all components.\n"
        "- Components: backlog pressure, aging open cases (>14 days), repeat requests, and resolution lag."
    )

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
    trend[["as_of_date", "frustration_index"]].rename(columns={"frustration_index": "score"}).assign(metric_label="Frustration Index")
]

if show_city_median:
    city_median_trend = (
        city_slice.groupby("as_of_date", as_index=False)["frustration_index"]
        .median()
        .rename(columns={"frustration_index": "score"})
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

st.subheader("Hotspots")
hotspot_slice = hotspots[hotspots["window_days"] == window].copy()
hotspot_slice["grain_value"] = hotspot_slice[grain].fillna("Unknown")
hotspot_slice = hotspot_slice[hotspot_slice["grain_value"] != "Unknown"]
geojson, geojson_path = load_boundary_geojson(grain)
boundary_join_key = None
if geojson is not None:
    boundary_join_key = detect_join_key(geojson, set(current_slice["grain_value"].astype(str)))

if geojson is not None and boundary_join_key is not None:
    choropleth_data = current_slice[["grain_value", "frustration_index"]].copy()
    fig_choropleth = px.choropleth_map(
        choropleth_data,
        geojson=geojson,
        locations="grain_value",
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
