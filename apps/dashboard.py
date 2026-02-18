from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    GRAIN_LABELS,
    baseline_period_bounds,
    build_service_change_table,
    driver_label,
    is_reserve_comm_plan,
    normalize_zipcode_columns,
    prior_week_average,
    read_metric_value,
    resolve_latest_metric_date,
    resolve_latest_slice_date,
    window_total_for_area,
)
from apps.dashboard_sections import render_citywide_context, render_focused_tab, render_global_tab
from apps.dashboard_state import build_dashboard_state


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

state = build_dashboard_state(
    frustration=frustration,
    neighborhood_daily_metrics=neighborhood_daily_metrics,
)
baseline_mode = state["baseline_mode"]
component_window = state["component_window"]
current_slice = state["current_slice"]
grain = state["grain"]
include_reserve = state["include_reserve"]
latest_date = state["latest_date"]
latest_date_display = state["latest_date_display"]
latest_daily_date = state["latest_daily_date"]
selected_value = state["selected_value"]
top_n = state["top_n"]
window = state["window"]

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
    render_global_tab(
        latest_date_display=latest_date_display,
        current_slice=current_slice,
        top_n=top_n,
        grain=grain,
        window=window,
        include_reserve=include_reserve,
        component_window=component_window,
        hotspots=hotspots,
        service_daily=service_daily,
        baseline_mode=baseline_mode,
        city_service_latest=city_service_latest,
    )
