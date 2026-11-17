from pathlib import Path

DAGS = Path(__file__).resolve().parents[3] / "dags"


def test_no_dag_uses_removed_airflow_2_arguments():
    for f in DAGS.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "schedule_interval" not in text, f.name
        assert "days_ago" not in text, f.name
