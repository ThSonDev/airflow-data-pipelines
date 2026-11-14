"""PostgreSQL access for the dashboard. Settings come from the POSTGRES_* env vars."""
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

DB_CONFIG = {
    "user": os.environ.get("POSTGRES_USER", "airflow"),
    "password": os.environ.get("POSTGRES_PASSWORD", "airflow"),
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "database": os.environ.get("POSTGRES_DB", "airflow"),
}


def _read_sql(query, params=None):
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    conn = engine.raw_connection()  # plain psycopg2 connection, closed manually
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        conn.close()


@st.cache_data(ttl=5)
def load_data():
    try:
        return _read_sql("SELECT * FROM absa_results ORDER BY RANDOM() LIMIT 300")
    except Exception as e:
        st.warning(f"Không thể kết nối đến PostgreSQL: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_retrain_results(topk=20):
    query = "SELECT * FROM retrain_results ORDER BY created_at DESC LIMIT %s"
    try:
        df = _read_sql(query, params=(topk,))
        if "mean_aspect_f1" in df.columns:  # best first; old rows have no value and go last
            df = df.sort_values("mean_aspect_f1", ascending=False, na_position="last")
        return df
    except Exception as e:
        st.warning(f"Không đọc được bảng retrain_results: {e}")
        return pd.DataFrame()
