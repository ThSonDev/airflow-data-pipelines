import logging
import os
import subprocess
from datetime import timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG

log = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
        dag_id="absa_streaming_lifecycle_demo",
        default_args=default_args,
        description="Orchestrate Kafka–Spark–PostgreSQL streaming lifecycle (1-Hour Demo)",
        schedule_interval=timedelta(hours=1),
        start_date=days_ago(1),
        catchup=False,
        dagrun_timeout=timedelta(minutes=55),  # must stay below the 1h schedule
        tags=["absa", "streaming", "kafka", "spark"],
) as dag:

    deploy_producer = BashOperator(
        task_id="deploy_producer",
        bash_command="timeout 45m python /opt/airflow/projects/absa_streaming/scripts/producer.py",
        retries=3,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=50),
        trigger_rule="all_done",
    )

    deploy_consumer = BashOperator(
        task_id="deploy_consumer",
        bash_command=(
        "timeout 45m spark-submit "
        "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.6.0 "
        "--py-files /opt/airflow/projects/absa_streaming/scripts/absa_core.py,"
        "/opt/airflow/projects/absa_streaming/scripts/inference.py,"
        "/opt/airflow/projects/absa_streaming/scripts/model_registry.py,"
        "/opt/airflow/projects/absa_streaming/scripts/sink.py,"
        "/opt/airflow/projects/absa_streaming/scripts/text_utils.py "
        "/opt/airflow/projects/absa_streaming/scripts/consumer_postgres_streaming.py"
        ),
        retries=5,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=50),
        trigger_rule="all_done",
    )

    def monitor_job():
        log.info("Checking streaming job checkpoint")
        path = "/opt/airflow/checkpoints/absa_streaming_checkpoint"
        if os.path.exists(path):
            size = subprocess.check_output(["du", "-sh", path]).decode().split()[0]
            log.info("Checkpoint exists (%s), job running normally", size)
        else:
            log.warning("No checkpoint found, job may have failed or was cleaned")

    monitor_stream = PythonOperator(
        task_id="monitor_stream",
        python_callable=monitor_job,
        trigger_rule="all_done",
    )

    cleanup_checkpoints = BashOperator(
        task_id="cleanup_checkpoints",
        bash_command=(
            "echo '[Cleanup] Removing old checkpoint...'; "
            "rm -rf /opt/airflow/checkpoints/absa_streaming_checkpoint || true; "
            "echo '[Cleanup] Done.'"
        ),
        trigger_rule="all_done",
    )

    [deploy_producer, deploy_consumer] >> monitor_stream >> cleanup_checkpoints
