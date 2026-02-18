import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard_data import (
    DB_PATH,
    load_city_daily_metrics,
    load_city_service_daily_metrics,
    load_frustration_data,
    load_hotspots_data,
    load_neighborhood_daily_metrics,
    load_neighborhood_service_daily_metrics,
    load_service_daily_data,
)

from apps.dashboard_helpers import (
    ALT_BOUNDARY_DIR,
    BASELINE_LABELS,
    BOUNDARY_DIR,
    BOUNDARY_FILE_CANDIDATES,
    COMPONENT_COLUMNS,
    COMPONENT_LABELS,
    GRAIN_LABELS,
    baseline_period_bounds,
    build_geojson_join_lookup,
    build_intervention_priority_table,
    build_service_change_table,
    build_window_request_rollup,
    detect_join_key,
    driver_label,
    filter_geojson_for_grain,
    geojson_property_is_numeric,
    is_reserve_comm_plan,
    load_boundary_geojson,
    normalize_join_value,
    normalize_zipcode_columns,
    prior_week_average,
    read_metric_value,
    resolve_latest_metric_date,
    resolve_latest_slice_date,
    summarize_busy_light_services,
    unknown_rate,
    window_total_for_area,
    window_unknown_rate,
)
from apps.dashboard_sections import render_citywide_context, render_focused_tab





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

citywide_daily = neighborhood_daily_metrics[
    (neighborhood_daily_metrics.get("grain_type") == grain)
    & (neighborhood_daily_metrics.get("grain_value") != "Unknown")
].copy()
render_citywide_context(citywide_daily=citywide_daily, grain_label=GRAIN_LABELS[grain])

focused_tab, global_tab = st.tabs(["Focused Area", "Global Landscape"])

with focused_tab:
    recency_focus_open_3d = read_metric_value(focus_daily_row, "opened_request_count_3d")
    recency_focus_open_7d = read_metric_value(focus_daily_row, "opened_request_count_7d")
    recency_focus_closed_3d = read_metric_value(focus_daily_row, "closed_request_count_3d")
    recency_focus_closed_7d = read_metric_value(focus_daily_row, "closed_request_count_7d")
    recency_city_open_3d = read_metric_value(city_daily_row, "opened_request_count_3d")
    recency_city_open_7d = read_metric_value(city_daily_row, "opened_request_count_7d")
    recency_city_closed_3d = read_metric_value(city_daily_row, "closed_request_count_3d")
    recency_city_closed_7d = read_metric_value(city_daily_row, "closed_request_count_7d")

    service_change = build_service_change_table(
        service_daily=service_daily,
        grain_type=grain,
        selected_value=selected_value,
        latest_date=latest_date,
        current_window_days=window,
        baseline_mode=baseline_mode,
    )

    current_start, _, baseline_start, baseline_end = baseline_period_bounds(latest_date, window, baseline_mode)
    baseline_components = component_trend[
        (component_trend["as_of_date"] >= baseline_start) & (component_trend["as_of_date"] <= baseline_end)
    ]

    render_focused_tab(
        selected_value=selected_value,
        window=window,
        component_window=component_window,
        latest_requests=latest_requests,
        request_delta_vs_prior_window=request_delta_vs_prior_window,
        top_driver=top_driver,
        selected_component_latest=selected_component_latest,
        recency_focus_open_3d=recency_focus_open_3d,
        recency_focus_open_7d=recency_focus_open_7d,
        recency_focus_closed_3d=recency_focus_closed_3d,
        recency_focus_closed_7d=recency_focus_closed_7d,
        recency_city_open_3d=recency_city_open_3d,
        recency_city_open_7d=recency_city_open_7d,
        recency_city_closed_3d=recency_city_closed_3d,
        recency_city_closed_7d=recency_city_closed_7d,
        service_change=service_change,
        baseline_mode=baseline_mode,
        baseline_components=baseline_components,
        component_trend=component_trend,
        focus_service_latest=focus_service_latest,
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
