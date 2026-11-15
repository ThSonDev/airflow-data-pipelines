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
    schedule=None,  # manual trigger: the CSV data is static, so a schedule would only retrain the same model
    max_active_runs=1,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['absa', 'cnn', 'retrain'],
) as dag:

    SCRIPTS_DIR = "/opt/airflow/projects/absa_streaming/scripts"
    # One folder per run. ts_nodash is path-safe (run_id has ":" and "+") and unique per run.
    RUN_TMP = "/opt/airflow/models/tmp/{{ ts_nodash }}"
    # append_env keeps POSTGRES_*, PATH, etc.; env alone would replace the whole environment
    script_env = {"env": {"TMP_DIR": RUN_TMP}, "append_env": True}

    preprocess_task = BashOperator(
        task_id="preprocess_data",
        bash_command=f'python3 {SCRIPTS_DIR}/preprocess.py',
        **script_env,
        execution_timeout=timedelta(minutes=50),
    )

    train_task = BashOperator(
        task_id="train_model",
        bash_command=f'python3 {SCRIPTS_DIR}/train.py',
        **script_env,
        execution_timeout=timedelta(minutes=130),
    )

    eval_task = BashOperator(
        task_id="evaluate_model",
        bash_command=f'python3 {SCRIPTS_DIR}/eval.py',
        **script_env,
        execution_timeout=timedelta(minutes=25),
    )

    save_task = BashOperator(
        task_id="check_and_deploy",
        bash_command=f'python3 {SCRIPTS_DIR}/save_postgres.py',
        **script_env,
        execution_timeout=timedelta(minutes=15),
    )

    cleanup_task = BashOperator(
        task_id="cleanup_tmp",
        bash_command=f'rm -rf {RUN_TMP}',
        trigger_rule="all_done",
    )

    preprocess_task >> train_task >> eval_task >> save_task >> cleanup_task
