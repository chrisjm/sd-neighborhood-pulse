from pathlib import Path

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


@st.cache_data(ttl=300)
def load_daily_metrics_data() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            """
            select *
            from main.fct_neighborhood_daily_metrics
            order by metric_date
            """
        ).fetchdf()


def unknown_rate(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    return float((df[column] == "Unknown").mean() * 100)


def driver_label(row: pd.Series) -> str:
    return COMPONENT_LABELS[row[COMPONENT_COLUMNS].astype(float).idxmax()]


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
daily_metrics = load_daily_metrics_data()

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

st.subheader(f"Current Frustration Index ({GRAIN_LABELS[grain]}, {window}-day)")
col1, col2 = st.columns([2, 1])

with col1:
    if PRESETS[preset_name] is None:
        st.caption("Custom mode enabled")
    else:
        st.caption(f"Preset mode: {preset_name}")
    top_ranked = current_slice.head(top_n).sort_values("frustration_index", ascending=True)
    fig_bar = px.bar(
        top_ranked,
        x="frustration_index",
        y="grain_value",
        orientation="h",
        color="frustration_index",
        color_continuous_scale="Reds",
        title=f"Top {top_n} areas by frustration index",
        labels={
            "frustration_index": "Frustration Index (0-100)",
            "grain_value": GRAIN_LABELS[grain],
        },
    )
    fig_bar.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, width="stretch")

with col2:
    st.metric("As of date", str(latest_date))
    st.metric("Rows in current cut", int(len(current_slice)))
    st.metric(
        "Median frustration index",
        f"{current_slice['frustration_index'].median():.2f}" if not current_slice.empty else "n/a",
    )
    st.metric(
        "Highest area now",
        str(current_slice.iloc[0]["grain_value"]) if not current_slice.empty else "n/a",
    )

with st.expander("How to read the frustration index", expanded=True):
    st.markdown(
        "- The **frustration index** is an equal-weight average of 4 components (0-100 each).\n"
        "- **Higher is worse** across all components.\n"
        "- Components: backlog pressure, aging open cases (>14 days), repeat requests, and resolution lag."
    )

st.subheader("Neighborhood decomposition")
selected_value = st.selectbox(
    f"Select a {GRAIN_LABELS[grain]}",
    options=current_slice["grain_value"].tolist()[:100],
)

selected_latest = current_slice[current_slice["grain_value"] == selected_value].iloc[0]
decomposition = pd.DataFrame(
    {
        "component": [COMPONENT_LABELS[col] for col in COMPONENT_COLUMNS],
        "score": [float(selected_latest[col]) for col in COMPONENT_COLUMNS],
    }
).sort_values("score", ascending=False)
fig_decomposition = px.bar(
    decomposition,
    x="component",
    y="score",
    title=f"Latest component breakdown: {selected_value}",
    color="score",
    color_continuous_scale="Reds",
    labels={"component": "Component", "score": "Component score (0-100)"},
)
fig_decomposition.update_layout(showlegend=False, yaxis_title="Component score (0-100)")
st.plotly_chart(fig_decomposition, width="stretch")

st.subheader("Trend")
trend = frustration[
    (frustration["window_days"] == window)
    & (frustration["grain_type"] == grain)
    & (frustration["grain_value"] == selected_value)
].sort_values("as_of_date")

if trend.empty:
    st.info("No trend points available for the selected area.")
    st.stop()

component_selection = st.multiselect(
    "Select components to overlay",
    options=COMPONENT_COLUMNS,
    default=COMPONENT_COLUMNS,
    format_func=lambda col: COMPONENT_LABELS[col],
)

trend_columns = ["frustration_index", *component_selection]
trend_long = trend[["as_of_date", *trend_columns]].melt(
    id_vars=["as_of_date"], value_vars=trend_columns, var_name="metric", value_name="score"
)
trend_long["metric_label"] = trend_long["metric"].replace(
    {"frustration_index": "Frustration Index", **COMPONENT_LABELS}
)

fig_trend = px.line(trend_long, x="as_of_date", y="score", color="metric_label", title=f"Index and components trend: {selected_value}")
fig_trend.update_layout(yaxis_title="Score (0-100)", xaxis_title="As of date", legend_title_text="Series")
st.plotly_chart(fig_trend, width="stretch")

st.subheader("Known peak drivers (Top N request-volume days)")
peak_n = st.slider("Peak days to inspect", min_value=3, max_value=15, value=7)
peak_days = trend.sort_values("request_count", ascending=False).head(peak_n).copy()
peak_days["driver_component"] = peak_days.apply(driver_label, axis=1)
peak_display = peak_days[
        [
            "as_of_date",
            "request_count",
            "frustration_index",
            "driver_component",
            "backlog_component",
            "aging_component",
            "repeat_component",
            "resolution_component",
        ]
    ].rename(
    columns={
        "as_of_date": "As of date",
        "request_count": "Requests",
        "frustration_index": "Frustration index",
        "driver_component": "Primary driver",
        "backlog_component": "Backlog",
        "aging_component": "Aging",
        "repeat_component": "Repeat",
        "resolution_component": "Resolution",
    }
)
st.dataframe(
    peak_display.style.format(
        {
            "Frustration index": "{:.2f}",
            "Backlog": "{:.2f}",
            "Aging": "{:.2f}",
            "Repeat": "{:.2f}",
            "Resolution": "{:.2f}",
        }
    ),
    width="stretch",
)

st.subheader("Data quality watch")
quality_slice = daily_metrics[daily_metrics["grain_type"] == grain].copy()
quality_slice["is_unknown"] = quality_slice["grain_value"] == "Unknown"
quality_total = quality_slice.groupby("metric_date", as_index=False)["request_count"].sum().rename(columns={"request_count": "request_count"})
quality_unknown = (
    quality_slice[quality_slice["is_unknown"]]
    .groupby("metric_date", as_index=False)["request_count"]
    .sum()
    .rename(columns={"request_count": "unknown_request_count"})
)
quality_trend = quality_total.merge(quality_unknown, on="metric_date", how="left")
quality_trend["unknown_request_count"] = quality_trend["unknown_request_count"].fillna(0)
quality_trend["unknown_request_rate_pct"] = (
    quality_trend["unknown_request_count"] / quality_trend["request_count"].where(quality_trend["request_count"] != 0, 1)
) * 100
fig_quality = px.line(
    quality_trend,
    x="metric_date",
    y="unknown_request_rate_pct",
    title=f"Unknown-rate trend ({GRAIN_LABELS[grain]})",
    labels={"metric_date": "Date", "unknown_request_rate_pct": "Unknown request rate (%)"},
)
fig_quality.update_layout(yaxis_ticksuffix="%")
st.plotly_chart(fig_quality, width="stretch")

st.subheader("Hotspots")
hotspot_slice = hotspots[hotspots["window_days"] == window].head(200)
show_technical = st.checkbox("Show technical hotspot fields", value=False)
base_hotspot_columns = [
    "comm_plan_name",
    "council_district",
    "zipcode",
    "request_count",
    "open_ratio_pct",
    "dominant_service_name",
]
technical_hotspot_columns = ["centroid_latitude", "centroid_longitude", "lat_bin", "lon_bin", "cluster_id"]
display_columns = base_hotspot_columns + technical_hotspot_columns if show_technical else base_hotspot_columns
hotspot_display = hotspot_slice[display_columns].rename(
    columns={
        "comm_plan_name": "Community Plan",
        "council_district": "Council District",
        "zipcode": "ZIP Code",
        "request_count": "Requests",
        "open_ratio_pct": "Open ratio (%)",
        "dominant_service_name": "Dominant service",
        "centroid_latitude": "Centroid lat",
        "centroid_longitude": "Centroid lon",
        "lat_bin": "Lat bin",
        "lon_bin": "Lon bin",
        "cluster_id": "Cluster ID",
    }
)
st.dataframe(
    hotspot_display.style.format({"Open ratio (%)": "{:.2f}"}),
    width="stretch",
)

st.info(
    "Manual refresh: run ingest + dbt run/test/snapshot from README before checking this dashboard. "
    "The app reads from data/db/city_health.duckdb."
)
