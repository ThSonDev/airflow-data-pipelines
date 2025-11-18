import streamlit as st
import pandas as pd
import psycopg2
import time
import json
from PIL import Image
import os

# Config
DB_HOST = "postgres"
DB_NAME = "airflow"
DB_USER = "airflow"
DB_PASS = "airflow"

def get_db_connection():
    return psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)

# Use the core st.cache_data to cache the RESULT of the query, not the connection.
@st.cache_data(ttl=1)
def load_latest_frame_resilient(table_name):
    # This logic is wrapped in a retry to survive Airflow's pg_terminate_backend.
    max_retries = 3
    
    for attempt in range(max_retries):
        conn = None
        try:
            # 1. Get a fresh connection attempt
            conn = get_db_connection()
            
            query = f"""
                SELECT frame_id, video_second, total_count, class_counts, image_path, annotation_json
                FROM {table_name} 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            # 2. Execute query and return DataFrame
            df = pd.read_sql(query, conn)
            
            # CRITICAL: Close the connection immediately so it doesn't hold a lock
            conn.close() 
            
            return df

        except psycopg2.InterfaceError as e:
            # This catches the "connection already closed" error.
            if attempt < max_retries - 1:
                print(f"InterfaceError detected for {table_name}. Attempting reconnect...")
                if conn: conn.close() # Ensure dead socket is closed before retrying
                time.sleep(0.1) 
                continue # Try the loop again
            else:
                # If retries fail, return empty data, but Streamlit keeps running.
                print(f"Failed to reconnect to {table_name} after {max_retries} attempts. {e}")
                if conn: conn.close()
                return pd.DataFrame()
        
        except Exception as e:
            # Catch all other DB errors (like missing tables if Spark failed)
            if conn: conn.close()
            return pd.DataFrame()
            
    return pd.DataFrame() # Should not be reached

st.set_page_config(page_title="YOLO Vehicle Counting", layout="wide")
st.title("🚗 Real-Time Vehicle Counter")

# Sidebar
refresh_rate = st.sidebar.slider("Refresh (s)", 1, 5, 2)
selected_video = st.sidebar.radio("Source", ["Video 1", "Video 2", "Video 3"])
table_map = {"Video 1": "yolo_results_video1", "Video 2": "yolo_results_video2", "Video 3": "yolo_results_video3"}
current_table = table_map[selected_video]

# Main Layout
placeholder = st.empty()

while True:
    with placeholder.container():
        df = load_latest_frame_resilient(current_table)
        
        if not df.empty:
            row = df.iloc[0]
            
            # Top Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            
            # Parse Counts
            try:
                counts = json.loads(row['class_counts'])
            except:
                counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            
            m1.metric("Total Vehicles", row['total_count'])
            m2.metric("Cars", counts.get('car', 0))
            m3.metric("Motos", counts.get('motorcycle', 0))
            m4.metric("Buses", counts.get('bus', 0))
            m5.metric("Trucks", counts.get('truck', 0))
            
            st.markdown("---")
            
            # Image & Details
            col_img, col_json = st.columns([2, 1])
            
            with col_img:
                st.subheader(f"Live View: {row['video_second']}s")
                if row['image_path'] and os.path.exists(row['image_path']):
                    # The image already has boxes drawn by YOLO (result.plot())
                    image = Image.open(row['image_path'])
                    st.image(image, use_column_width=True, caption=f"Frame ID: {row['frame_id']}")
                else:
                    st.warning("Image file not found.")

            with col_json:
                st.subheader("Detection Data")
                st.json(row['annotation_json']) # Shows the raw box coordinates

        else:
            st.info("Waiting for data...")
            
    time.sleep(refresh_rate)