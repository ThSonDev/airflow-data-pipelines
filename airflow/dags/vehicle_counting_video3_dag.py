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

dag_id='yolo_pipeline_video3'
video_path='/opt/airflow/projects/vehicle_counting/data/video3.mp4'
kafka_topic='video3_topic'
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
        timeout 15m python {PRODUCER_SCRIPT_PATH} \
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
        execution_timeout=timedelta(minutes=15)
    )

    # 3. Cleanup Task
    cleanup_task = BashOperator(
        task_id='cleanup_task',
        bash_command=(
            "echo 'Cleaning up image files and database table...';"
            # 1. Delete image files saved by the Spark worker
            "rm -rf /opt/airflow/static/processed_frames/video3_*;"
            
            # 2. TRUNCATE table using docker exec
            #"docker exec airflow-postgres-1 psql -U airflow -d airflow -c "
            #"'TRUNCATE TABLE yolo_results_video3 RESTART IDENTITY;'"
            
            "echo 'Cleanup complete.'"
        ),
        execution_timeout=timedelta(minutes=1), # Ensure cleanup doesn't hang
        trigger_rule='all_done'
    )

    [producer_task, consumer_task] >> cleanup_task