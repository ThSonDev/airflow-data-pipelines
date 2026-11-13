# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# Dashboard for ABSA results stored in PostgreSQL. It refreshes every 5 seconds.
import pandas as pd
import plotly.express as px
import streamlit as st
from db import load_data, load_retrain_results
from streamlit_autorefresh import st_autorefresh

ASPECTS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
SENTIMENTS = ["POS", "NEU", "NEG"]
COLORS = {"POS": "#33cc33", "NEU": "#cccc00", "NEG": "#ff5050"}

st.set_page_config(page_title="ABSA Streaming Dashboard", layout="wide")
st.title("Real-time ABSA Social Listening Dashboard")
st.caption("Minh hoạ pipeline Kafka -> Spark -> PostgreSQL -> Streamlit (CNPM – UIT)")
st_autorefresh(interval=5 * 1000, limit=None, key="auto_refresh")

df = load_data()

if df.empty:
    st.warning("Chưa có dữ liệu trong bảng `absa_results`. Hãy đảm bảo producer và consumer đang chạy.")
else:
    st.subheader("Dữ liệu gần đây")
    st.dataframe(df.tail(10), use_container_width=True)

    st.subheader("Thống kê cảm xúc theo khía cạnh")
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
        st.markdown("#### Biểu đồ tổng hợp cảm xúc theo khía cạnh")
        fig_bar = px.bar(
            df_stats,
            x="Aspect", y="Count", color="Sentiment",
            color_discrete_map=COLORS,
            barmode="group", text_auto=True
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

    with col2:
        st.markdown("#### Tỉ lệ cảm xúc tích cực / trung tính / tiêu cực")
        df_total = df_stats.groupby("Sentiment")["Count"].sum().reset_index()
        fig_pie = px.pie(
            df_total, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map=COLORS,
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    st.markdown("---")
    st.subheader("Danh sách model retrain tốt (theo Macro-F1)")

    top_models = load_retrain_results(topk=30)

    if top_models.empty:
        st.info("Chưa có log retrain nào trong bảng `retrain_results`.")
    else:
        top_models = top_models.rename(columns={
            "macro_f1": "Macro F1",
            "balanced_acc": "Balanced Acc",
            "model_name": "Model Name",
            "created_at": "Created At",
            "id": "ID"
        })
        for c in ["Macro F1", "Balanced Acc"]:
            if c in top_models.columns:
                top_models[c] = top_models[c].astype(float).round(4)
        st.dataframe(top_models, use_container_width=True, hide_index=True)
