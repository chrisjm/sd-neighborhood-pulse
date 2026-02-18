from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

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

    focus_areas = current_slice.sort_values("request_count", ascending=False).head(max_rows)
    hotspot_summary = pd.DataFrame()
    if not hotspot_slice.empty:
        hotspot_summary = (
            hotspot_slice.groupby("grain_value", as_index=False)
            .agg(
                dominant_service=(
                    "dominant_service_name", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
                ),
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
            service_name = "Unknown"
        else:
            top_service = service_change.iloc[0]
            service_name = str(top_service["service_name"])
            trend_pct = top_service["pct_change"]

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
