from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_helpers import (
    BASELINE_LABELS,
    GRAIN_LABELS,
    build_window_request_rollup,
    is_reserve_comm_plan,
    resolve_latest_metric_date,
    unknown_rate,
    window_unknown_rate,
)


def build_dashboard_state(
    frustration: pd.DataFrame,
    neighborhood_daily_metrics: pd.DataFrame,
) -> dict[str, object]:
    latest_date = frustration["as_of_date"].max()
    latest_date_display = pd.to_datetime(latest_date).date().isoformat()

    st.sidebar.header("Filters")
    window = st.sidebar.selectbox("Window (days)", [7, 30, 90], index=0)
    grain = st.sidebar.selectbox(
        "Neighborhood grain",
        ["comm_plan_name", "council_district", "zipcode"],
        index=0,
    )
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
        current_slice_all = build_window_request_rollup(
            neighborhood_daily_metrics,
            grain,
            latest_daily_date,
            window,
        )
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
        current_unknown_rate = window_unknown_rate(
            neighborhood_daily_metrics,
            grain,
            latest_daily_date,
            window,
        )
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

    return {
        "baseline_mode": baseline_mode,
        "component_window": component_window,
        "current_slice": current_slice,
        "grain": grain,
        "include_reserve": include_reserve,
        "latest_date": latest_date,
        "latest_date_display": latest_date_display,
        "latest_daily_date": latest_daily_date,
        "selected_value": selected_value,
        "top_n": top_n,
        "window": window,
    }
