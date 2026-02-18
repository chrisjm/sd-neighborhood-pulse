from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard_helpers import (
    ALT_BOUNDARY_DIR,
    BASELINE_LABELS,
    BOUNDARY_DIR,
    COMPONENT_COLUMNS,
    COMPONENT_LABELS,
    GRAIN_LABELS,
    build_geojson_join_lookup,
    build_intervention_priority_table,
    detect_join_key,
    filter_geojson_for_grain,
    geojson_property_is_numeric,
    load_boundary_geojson,
    normalize_join_value,
    summarize_busy_light_services,
)


def render_citywide_context(citywide_daily: pd.DataFrame, grain_label: str) -> None:
    st.subheader("Citywide context")
    if citywide_daily.empty:
        st.info("Citywide trend unavailable for this grain.")
        return

    citywide_daily = citywide_daily.sort_values("metric_date")
    citywide_summary = (
        citywide_daily.groupby("metric_date", as_index=False)["request_count"]
        .sum()
        .rename(columns={"request_count": "requests"})
    )
    citywide_summary = citywide_summary.tail(365)
    fig_city = px.line(
        citywide_summary,
        x="metric_date",
        y="requests",
        title=f"Citywide request volume trend ({grain_label})",
        labels={"metric_date": "Date", "requests": "Total requests"},
    )
    fig_city.update_layout(yaxis_title="Requests")
    st.plotly_chart(fig_city, width="stretch")


def render_focused_tab(
    selected_value: str,
    window: int,
    component_window: int,
    latest_requests: float,
    request_delta_vs_prior_window: float | None,
    top_driver: str,
    selected_component_latest: pd.DataFrame,
    recency_focus_open_3d: float | None,
    recency_focus_open_7d: float | None,
    recency_focus_closed_3d: float | None,
    recency_focus_closed_7d: float | None,
    recency_city_open_3d: float | None,
    recency_city_open_7d: float | None,
    recency_city_closed_3d: float | None,
    recency_city_closed_7d: float | None,
    service_change: pd.DataFrame,
    baseline_mode: str,
    baseline_components: pd.DataFrame,
    component_trend: pd.DataFrame,
    focus_service_latest: pd.DataFrame,
) -> None:
    st.subheader("What changed this week")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Selected area", selected_value)
    repeat_value = _read_component_value(selected_component_latest, "repeat_component")
    resolution_value = _read_component_value(selected_component_latest, "resolution_component")
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
            f"{request_delta_vs_prior_window:+.1f} vs prior window"
            if request_delta_vs_prior_window is not None
            else None
        ),
    )

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
        weekly_direction_text = (
            "increasing week-over-week" if request_delta_vs_prior_window >= 0 else "decreasing week-over-week"
        )
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
            }
        )
        st.dataframe(action_table, width="stretch", hide_index=True)

    if not baseline_components.empty:
        baseline_component_values = baseline_components[COMPONENT_COLUMNS].mean()
        component_delta = pd.DataFrame(
            {
                "component": COMPONENT_COLUMNS,
                "current_score": [
                    float(selected_component_latest.iloc[0].get(column, 0.0)) for column in COMPONENT_COLUMNS
                ],
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


def render_global_tab(
    latest_date_display: str,
    current_slice: pd.DataFrame,
    top_n: int,
    grain: str,
    window: int,
    include_reserve: bool,
    component_window: int,
    hotspots: pd.DataFrame,
    service_daily: pd.DataFrame,
    baseline_mode: str,
    city_service_latest: pd.DataFrame,
) -> None:
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
        latest_date=pd.to_datetime(latest_date_display),
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
                choropleth_data["join_location"] = choropleth_data["grain_geo_value"].fillna(
                    choropleth_data["grain_value"]
                )
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


def _read_component_value(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    value = df.iloc[0][column]
    if pd.isna(value):
        return None
    return float(value)
