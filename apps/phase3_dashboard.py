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
frustration = normalize_zipcode_columns(frustration)
hotspots = normalize_zipcode_columns(hotspots)

if frustration.empty:
    st.warning("No frustration index data available yet. Run dbt models first.")
    st.stop()

latest_date = frustration["as_of_date"].max()
latest_date_display = pd.to_datetime(latest_date).date().isoformat()
st.sidebar.header("Filters")
window = st.sidebar.selectbox("Window (days)", [30, 90], index=0)
grain = st.sidebar.selectbox("Neighborhood grain", ["comm_plan_name", "council_district", "zipcode"], index=0)
st.sidebar.caption("Showing all areas for the selected grain and window.")
include_reserve = False
if grain == "comm_plan_name":
    include_reserve = st.sidebar.checkbox("Include Reserve (power users)", value=False)
    if include_reserve:
        st.sidebar.caption("Including Reserve area in community plan views.")
    else:
        st.sidebar.caption("Reserve is excluded by default. Enable the checkbox to include it.")

current_slice_all = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["as_of_date"] == latest_date)
].sort_values("frustration_index", ascending=False)
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
city_median_latest = float(current_slice["frustration_index"].median())
latest_index = float(selected_latest["frustration_index"])
latest_requests = float(selected_latest["request_count"])
prior_week_index_avg = prior_week_average(trend, "frustration_index", latest_date)
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
        current_slice.sort_values("frustration_index", ascending=False)
        .head(top_n)
        .sort_values("frustration_index", ascending=True)
    )
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
    chart_height = min(1400, max(560, int(len(top_ranked) * 28)))
    fig_bar.update_layout(height=chart_height, coloraxis_showscale=False)
    fig_bar.update_yaxes(type="category")
    st.plotly_chart(fig_bar, width="stretch")

    with st.expander("How to read the frustration index", expanded=False):
        st.markdown(
            "- The **frustration index** is an equal-weight average of 4 components (0-100 each).\n"
            "- **Higher is worse** across all components.\n"
            "- Components: backlog pressure, aging open cases (>14 days), repeat requests, and resolution lag."
        )

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
