from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Base paths - Adjust based on your Docker mapping
PRODUCER_SCRIPT_PATH = "/opt/airflow/projects/vehicle_counting/scripts/producer.py"
CONSUMER_SCRIPT_PATH = "/opt/airflow/projects/vehicle_counting/scripts/consumer_yolo.py"

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag_id='yolo_pipeline_video1'
video_path='/opt/airflow/projects/vehicle_counting/data/video1.mp4'
kafka_topic='video1_topic'
default_args=default_args

    
with DAG(
        dag_id=dag_id,
        default_args=default_args,
        description=f'YOLO Pipeline for {video_path}',
        schedule_interval=None, # Trigger manually
        start_date=days_ago(1),
        catchup=False,
        tags=['yolo', 'video', 'spark']
) as dag:
    # 1. Producer Task
    producer_task = BashOperator(
        task_id='producer_task',
        bash_command=f"""
        timeout 3m python {PRODUCER_SCRIPT_PATH} \
        --video_path {video_path} \
        --topic {kafka_topic}
        """
    )

    # 2. Consumer Task
    consumer_task = BashOperator(
        task_id='consumer_task',
        bash_command=f"""
        spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.6.0 \
        {CONSUMER_SCRIPT_PATH} \
        --topics {kafka_topic}
        """,
        execution_timeout=timedelta(minutes=3)
    )

    # 3. Cleanup Task
    cleanup_task = BashOperator(
        task_id='cleanup_task',
        bash_command=(
            "echo 'Cleaning up image files and killing blocking DB connections...';"
            
            # 1. Kill ALL connections to the 'airflow' database EXCEPT the current one (to avoid killing psql itself)
            # This handles Streamlit and any zombie connections.
            #"docker exec postgres psql -U airflow -d airflow -c "
            #"'SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = ''airflow'' AND pid != pg_backend_pid();';"
            
            # 2. TRUNCATE the table and reset the ID counter
            #"docker exec postgres psql -U airflow -d airflow -c "
            #"'TRUNCATE TABLE yolo_results_video1 RESTART IDENTITY;';"
            
            # 3. Delete image files saved by the Spark worker
            "rm -rf /opt/airflow/static/processed_frames/video1_*;"
            
            "echo 'Cleanup complete.'"
        ),
        execution_timeout=timedelta(minutes=1),
        trigger_rule='all_done'
    )

    [producer_task, consumer_task] >> cleanup_task