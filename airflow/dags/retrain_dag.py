from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# =============================================================================
# AIRFLOW DAG CONFIGURATION
# =============================================================================
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
    schedule_interval='@hourly',  # Run hourly
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['absa', 'cnn', 'retrain'],
) as dag:

    # Paths
    SCRIPTS_DIR = "/opt/airflow/projects/absa_streaming/scripts"
    
    # 1. Preprocess: CSV -> Tensors/Vocab in tmp
    # FIX: "timeout" goes first, then "python3", then the script path (no inner quotes)
    preprocess_task = BashOperator(
        task_id="preprocess_data",
        bash_command=f'timeout 45m python3 {SCRIPTS_DIR}/preprocess.py',
        execution_timeout=timedelta(minutes=50),
    )

    # 2. Train: Tensors -> model_temp.pth in tmp
    # FIX: "timeout" goes first
    train_task = BashOperator(
        task_id="train_model",
        bash_command=f'timeout 120m python3 {SCRIPTS_DIR}/train.py',
        execution_timeout=timedelta(minutes=130),
    )

    # 3. Evaluate: model_temp.pth -> metrics.json in tmp
    # FIX: "timeout" goes first
    eval_task = BashOperator(
        task_id="evaluate_model",
        bash_command=f'timeout 20m python3 {SCRIPTS_DIR}/eval.py',
        execution_timeout=timedelta(minutes=25),
    )

    # 4. Compare & Save: metrics.json vs DB -> Move file or Ignore
    # FIX: "timeout" goes first
    save_task = BashOperator(
        task_id="check_and_deploy",
        bash_command=f'timeout 10m python3 {SCRIPTS_DIR}/save_postgres.py',
        execution_timeout=timedelta(minutes=15),
    )

    # 5. Cleanup: Delete tmp folder
    cleanup_task = BashOperator(
        task_id="cleanup_tmp",
        bash_command='rm -rf /opt/airflow/models/tmp/*',
        trigger_rule="all_done",  # Run even if previous tasks fail
    )

    # Workflow
    preprocess_task >> train_task >> eval_task >> save_task >> cleanup_task