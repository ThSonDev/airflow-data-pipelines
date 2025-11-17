from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import torch

def check_gpu():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("Current device:", torch.cuda.current_device())
    else:
        print("!!! No GPU detected, running on CPU only.")

with DAG(
    dag_id="check_gpu_dag",
    description="Test GPU availability inside Airflow task container",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["gpu", "test"]
) as dag:

    test_gpu_task = PythonOperator(
        task_id="check_gpu_task",
        python_callable=check_gpu,
    )

    test_gpu_task
