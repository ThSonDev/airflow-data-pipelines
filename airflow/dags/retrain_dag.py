from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'absa_retrain_pipeline',
    default_args=default_args,
    description='Retrain TextCNN ABSA model and deploy if better',
    schedule_interval='@hourly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['absa', 'cnn', 'retrain'],
) as dag:

    SCRIPTS_DIR = "/opt/airflow/projects/absa_streaming/scripts"
    
    preprocess_task = BashOperator(
        task_id="preprocess_data",
        bash_command=f'timeout 45m python3 {SCRIPTS_DIR}/preprocess.py',
        execution_timeout=timedelta(minutes=50),
    )

    train_task = BashOperator(
        task_id="train_model",
        bash_command=f'timeout 120m python3 {SCRIPTS_DIR}/train.py',
        execution_timeout=timedelta(minutes=130),
    )

    eval_task = BashOperator(
        task_id="evaluate_model",
        bash_command=f'timeout 20m python3 {SCRIPTS_DIR}/eval.py',
        execution_timeout=timedelta(minutes=25),
    )

    save_task = BashOperator(
        task_id="check_and_deploy",
        bash_command=f'timeout 10m python3 {SCRIPTS_DIR}/save_postgres.py',
        execution_timeout=timedelta(minutes=15),
    )

    cleanup_task = BashOperator(
        task_id="cleanup_tmp",
        bash_command='rm -rf /opt/airflow/models/tmp/*',
        trigger_rule="all_done",
    )

    preprocess_task >> train_task >> eval_task >> save_task >> cleanup_task
