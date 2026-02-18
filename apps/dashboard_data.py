from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

DB_PATH = Path("data/db/city_health.duckdb")


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
def load_neighborhood_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_neighborhood_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_city_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_city_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_neighborhood_service_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_neighborhood_service_daily_metrics
                """
            ).fetchdf()
        except duckdb.Error:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def load_city_service_daily_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        try:
            return conn.execute(
                """
                select *
                from main.fct_city_service_daily_metrics
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
