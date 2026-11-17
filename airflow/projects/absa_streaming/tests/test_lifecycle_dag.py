import runpy
import sys
import types
from pathlib import Path

DAG_FILE = Path(__file__).resolve().parents[3] / "dags" / "absa_streaming_lifecycle_dag.py"
MODULES = ["absa_core", "inference", "model_registry", "sink", "text_utils"]


def _load_tasks(monkeypatch):
    """Run the DAG file with fake airflow classes (airflow is not installed in the dev venv)."""
    tasks = []

    class FakeDAG:
        def __init__(self, dag_id, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            tasks.append(self)

        def __rshift__(self, other):
            return other

        def __rrshift__(self, other):
            return self

    for name, attrs in {
        "airflow": {},
        "airflow.sdk": {"DAG": FakeDAG},
        "airflow.providers": {},
        "airflow.providers.standard": {},
        "airflow.providers.standard.operators": {},
        "airflow.providers.standard.operators.bash": {"BashOperator": FakeTask},
        "airflow.providers.standard.operators.python": {"PythonOperator": FakeTask},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    runpy.run_path(str(DAG_FILE))
    return {t.kwargs["task_id"]: t.kwargs for t in tasks}


def test_no_task_deletes_the_checkpoint(monkeypatch):
    tasks = _load_tasks(monkeypatch)
    assert "cleanup_checkpoints" not in tasks
    assert all("rm -rf" not in str(t.get("bash_command", "")) for t in tasks.values())


def test_consumer_lists_all_py_files(monkeypatch):
    cmd = _load_tasks(monkeypatch)["deploy_consumer"]["bash_command"]
    for m in MODULES:
        assert f"/scripts/{m}.py" in cmd


def test_no_timeout_wrapper(monkeypatch):
    tasks = _load_tasks(monkeypatch)
    assert all("timeout " not in str(t.get("bash_command", "")) for t in tasks.values())
