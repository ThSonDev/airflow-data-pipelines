# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# streamlit_app.py
# ======================================
# Dashboard hiển thị kết quả phân tích cảm xúc (ABSA) từ PostgreSQL
# và tự động cập nhật theo thời gian thực.

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import time
import plotly.express as px

# ------------------------
# Cấu hình kết nối PostgreSQL
# ------------------------
DB_CONFIG = {
    "user": "airflow",
    "password": "airflow",
    "host": "postgres",  # dùng tên service Docker
    "port": 5432,
    "database": "airflow"
}

ASPECTS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
SENTIMENTS = ["POS", "NEU", "NEG"]

# ------------------------
# Hàm load dữ liệu an toàn (dùng raw_connection)
# ------------------------
@st.cache_data(ttl=5)
def load_data():
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    try:
        conn = engine.raw_connection()  # Lấy psycopg2 connection thực
        try:
            df = pd.read_sql("SELECT * FROM absa_results ORDER BY RANDOM() LIMIT 300", conn)
        finally:
            conn.close()  # đảm bảo đóng kết nối
        return df
    except Exception as e:
        st.warning(f"⚠️ Không thể kết nối đến PostgreSQL: {e}")
        return pd.DataFrame()

# ------------------------
# Giao diện chính
# ------------------------
st.set_page_config(page_title="ABSA Streaming Dashboard", layout="wide")
st.title("📊 Real-time ABSA Social Listening Dashboard")
st.caption("Minh hoạ pipeline Kafka → Spark → PostgreSQL → Streamlit (CNPM – UIT)")

# ========================
# ✅ Auto-refresh mỗi 5 giây
# ========================
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5 * 1000, limit=None, key="auto_refresh")

# ------------------------
# Hiển thị dữ liệu
# ------------------------
df = load_data()

if df.empty:
    st.warning("⏳ Chưa có dữ liệu trong bảng `absa_results`. Hãy đảm bảo producer và consumer đang chạy.")
else:
    st.subheader("📝 Dữ liệu gần đây")
    st.dataframe(df.tail(10), use_container_width=True)

    st.subheader("📈 Thống kê cảm xúc theo khía cạnh")
    aspect_counts = []
    for asp in ASPECTS:
        if asp not in df.columns:
            continue
        counts = df[asp].value_counts().reindex(SENTIMENTS, fill_value=0)
        for sent, cnt in counts.items():
            aspect_counts.append({"Aspect": asp, "Sentiment": sent, "Count": cnt})
    df_stats = pd.DataFrame(aspect_counts)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔹 Biểu đồ tổng hợp cảm xúc theo khía cạnh")
        fig_bar = px.bar(
            df_stats,
            x="Aspect", y="Count", color="Sentiment",
            color_discrete_map={"POS": "#33cc33", "NEU": "#cccc00", "NEG": "#ff5050"},
            barmode="group", text_auto=True
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

    with col2:
        st.markdown("#### 🔹 Tỉ lệ cảm xúc tích cực / trung tính / tiêu cực")
        df_total = df_stats.groupby("Sentiment")["Count"].sum().reset_index()
        fig_pie = px.pie(
            df_total, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map={"POS": "#33cc33", "NEU": "#cccc00", "NEG": "#ff5050"},
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    st.markdown("---")
    st.subheader("🏆 Danh sách model retrain tốt (theo Macro-F1)")

    @st.cache_data(ttl=5)
    def load_retrain_results(topk=20):  
        try:
            engine = create_engine(
                f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
                f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )
            conn = engine.raw_connection()
            try:
                q = """
                SELECT model_name, epoch, acc, f1_macro, precision_macro, recall_macro, loss, created_at
                FROM retrain_results
                ORDER BY f1_macro DESC NULLS LAST, created_at DESC
                LIMIT %s
                """
                dfm = pd.read_sql(q, conn, params=(topk,))
            finally:
                conn.close()
            return dfm
        except Exception as e:
            st.warning(f"⚠️ Không đọc được bảng retrain_results: {e}")
            return pd.DataFrame()

top_models = load_retrain_results(topk=30)

if top_models.empty:
    st.info("Chưa có log retrain nào trong bảng `retrain_results`.")
else:
    # Hiển thị bảng gọn gàng, làm tròn các số
    num_cols = ["acc", "f1_macro", "precision_macro", "recall_macro", "loss"]
    for c in num_cols:
        if c in top_models.columns:
            top_models[c] = top_models[c].astype(float).round(4)

    st.dataframe(top_models, use_container_width=True)