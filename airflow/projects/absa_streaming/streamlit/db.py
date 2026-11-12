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
    query = """
        SELECT id, model_name, macro_f1, balanced_acc, created_at
        FROM retrain_results
        ORDER BY macro_f1 DESC NULLS LAST, created_at DESC
        LIMIT %s
    """
    try:
        return _read_sql(query, params=(topk,))
    except Exception as e:
        st.warning(f"Không đọc được bảng retrain_results: {e}")
        return pd.DataFrame()
