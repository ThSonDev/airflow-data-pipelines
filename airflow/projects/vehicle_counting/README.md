# YOLO Real-Time Vehicle Counting Pipeline

This project implements a real-time vehicle detection and counting pipeline using **Kafka → Spark Structured Streaming → PostgreSQL → Airflow → Streamlit**.
The system simulates a multi-camera traffic monitoring workflow where frames are streamed into Kafka, processed by YOLO, stored in a database, and displayed in a live dashboard.

## 1. Overview

This pipeline processes videos from multiple sources (3 cameras).
It includes:

* A **Kafka producer** that streams frames from video files at 1 FPS 
* A **Spark Structured Streaming consumer** that runs YOLO inference and writes results to PostgreSQL 
* A **PostgreSQL** database storing bounding boxes, counts, and annotated frame paths
* A **Streamlit dashboard** that shows live detections and metrics with auto-refresh 
* **Airflow DAGs** for managing ingestion, processing, cleanup, and multi-video workflows
* Fully containerized environment using Docker

This setup reproduces a realistic real-time analytics scenario for traffic monitoring.

## 2. Architecture

Video → Kafka → Spark Structured Streaming → PostgreSQL → Streamlit Dashboard

Three DAGs are provided: `video1`, `video2`, and `video3`.

### Source Code

- **Producer**  
  Handles video frame extraction and pushes messages to Kafka.  
  [`producer.py`](airflow/projects/vehicle_counting/scripts/producer.py)

- **Consumer (YOLO Inference)**  
  Consumes frames, performs object detection, saves results to Postgres.  
  [`consumer_yolo.py`](airflow/projects/vehicle_counting/scripts/consumer_yolo.py)

- **Streamlit Dashboard**  
  Real-time vehicle counting visualization.  
  [`app.py`](airflow/projects/vehicle_counting/streamlit/app.py)

## 3. Features

### Real-Time Frame Streaming

* Extracts frames at 1 FPS from each video
* Resizes and encodes frames to Base64 JSON messages
* Sends frames to Kafka topics (`video1_topic`, `video2_topic`, `video3_topic`)
  (Producer logic: frame capture → resize → JPEG encode → Base64 → Kafka produce) 

### YOLO-Based Detection with Spark

The Spark consumer:

* Reads frames from Kafka
* Decodes Base64 → image array
* Runs YOLO detection (`consumer_yolo.py`)
* Extracts bounding boxes, class counts, total vehicle count
* Saves results to PostgreSQL with image paths

### PostgreSQL Storage

Each video has a dedicated table (`yolo_results_video1`, `yolo_results_video2`, `yolo_results_video3`).

Stored fields include:

* `frame_id`, `video_second`
* `total_count`
* `class_counts` (JSON)
* `annotation_json`
* `image_path` (YOLO-annotated image)
* `created_at`

### Streamlit Live Dashboard

Dashboard features:

* Vehicle counts (car, motorcycle, bus, truck)
* Total vehicles per frame
* Live frame preview with YOLO bounding boxes
* Annotated metadata viewer
* Auto-refresh (1–5 seconds configurable)

![Streamlit Dashboard](images/dashboard.gif)

## 4. Airflow Management

### Main DAGs (per video)

Each DAG runs:

* Producer → Consumer → Cleanup
* Timeout, retries, and error handling

Other similar DAGs:

* `yolo_pipeline_video2`
* `yolo_pipeline_video3`

### Cleanup Logic

After each run:

* Remove annotated frame files
* Reset or clear PostgreSQL table
* Prevent leftover worker files from filling storage
* (Database cleanup code exists but is commented out in the DAG.)

## 5. Project Structure

```
airflow/
│── dags/
│     ├── yolo_pipeline_video1_dag.py
│     ├── yolo_pipeline_video2_dag.py
│     └── yolo_pipeline_video3_dag.py
│── projects/vehicle_counting/
│     ├── scripts/
│     │     ├── producer.py
│     │     ├── consumer_yolo.py
│     ├── streamlit/app.py
│     ├── data/
│     │     ├── video1.mp4
│     │     ├── video2.mp4
│     │     └── video3.mp4
│── static/processed_frames/
│     ├── video1_*.jpg
│     ├── video2_*.jpg
│     └── video3_*.jpg
```
