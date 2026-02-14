from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/db/city_health.duckdb")


@st.cache_data(ttl=300)
def load_frustration_data() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            """
            select *
            from main.fct_neighborhood_frustration_index
            where grain_value <> 'Unknown'
            """
        ).fetchdf()


@st.cache_data(ttl=300)
def load_hotspots_data() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            """
            select *
            from main.fct_request_hotspots
            where comm_plan_name <> 'Unknown'
            order by window_days, request_count desc
            """
        ).fetchdf()


st.set_page_config(page_title="SD Neighborhood Pulse", layout="wide")
st.title("San Diego Neighborhood Pulse")
st.caption("Phase 3 v1: frustration index + hotspot monitoring")

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

if PRESETS[preset_name] is not None:
    share_hashtags = "#SanDiego #NeighborhoodPulse #CivicData"
    st.sidebar.markdown("### Share-ready caption")
    st.sidebar.code(
        f"{preset_name} | As of {latest_date} | Source: Get It Done\n{share_hashtags}",
        language="text",
    )

current_slice = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["as_of_date"] == latest_date)
].sort_values("frustration_index", ascending=False)

if current_slice.empty:
    st.warning(
        f"No rows available for {grain} at {window}-day window on {latest_date}. "
        "Try a different preset or switch to Custom mode."
    )
    st.stop()

st.subheader(f"Current Frustration Index ({grain}, {window}-day)")
col1, col2 = st.columns([2, 1])

with col1:
    if PRESETS[preset_name] is None:
        st.caption("Custom mode enabled")
    else:
        st.caption(f"Preset mode: {preset_name}")
    fig_bar = px.bar(
        current_slice.head(top_n),
        x="frustration_index",
        y="grain_value",
        orientation="h",
        color="frustration_index",
        color_continuous_scale="Reds",
        title=f"Top {top_n} areas by frustration index",
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig_bar, width="stretch")

with col2:
    st.metric("As of date", str(latest_date))
    st.metric("Rows in current cut", int(len(current_slice)))
    st.metric(
        "Median frustration index",
        f"{current_slice['frustration_index'].median():.2f}" if not current_slice.empty else "n/a",
    )

st.subheader("Trend")
selected_value = st.selectbox(
    f"Select a {grain}",
    options=current_slice["grain_value"].tolist()[:100],
)
trend = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["grain_value"] == selected_value)
].sort_values("as_of_date")

if trend.empty:
    st.info("No trend points available for the selected area.")
    st.stop()

fig_trend = px.line(
    trend,
    x="as_of_date",
    y=["frustration_index", "backlog_component", "aging_component", "repeat_component", "resolution_component"],
    title=f"Index and components trend: {selected_value}",
)
st.plotly_chart(fig_trend, width="stretch")

st.subheader("Hotspots")
hotspot_slice = hotspots[hotspots["window_days"] == window].head(200)
st.dataframe(
    hotspot_slice[
        [
            "comm_plan_name",
            "council_district",
            "zipcode",
            "request_count",
            "open_ratio_pct",
            "dominant_service_name",
            "centroid_latitude",
            "centroid_longitude",
        ]
    ],
    width="stretch",
)

st.info(
    "Manual refresh: run ingest + dbt run/test/snapshot from README before checking this dashboard. "
    "The app reads from data/db/city_health.duckdb."
)
