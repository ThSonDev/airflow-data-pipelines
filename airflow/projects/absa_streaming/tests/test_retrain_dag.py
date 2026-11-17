import runpy
import sys
import types
from pathlib import Path

DAG_FILE = Path(__file__).resolve().parents[3] / "dags" / "retrain_dag.py"
RUN_TMP = "/opt/airflow/models/tmp/{{ run_id | replace(':', '') | replace('+', '') }}"


def _load_dag(monkeypatch):
    """Run the DAG file with fake airflow classes (airflow is not installed in the dev venv)."""
    tasks, dags = [], []

    class FakeDAG:
        def __init__(self, dag_id, **kwargs):
            self.kwargs = kwargs
            dags.append(self)

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

    for name, attrs in {
        "airflow": {},
        "airflow.sdk": {"DAG": FakeDAG},
        "airflow.providers": {},
        "airflow.providers.standard": {},
        "airflow.providers.standard.operators": {},
        "airflow.providers.standard.operators.bash": {"BashOperator": FakeTask},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    runpy.run_path(str(DAG_FILE))
    return dags[0], {t.kwargs["task_id"]: t.kwargs for t in tasks}


def test_dag_settings(monkeypatch):
    dag, _ = _load_dag(monkeypatch)
    assert dag.kwargs["schedule"] is None
    assert dag.kwargs["max_active_runs"] == 1


def test_script_tasks_use_a_per_run_tmp_dir(monkeypatch):
    _, tasks = _load_dag(monkeypatch)
    script_tasks = [t for t in tasks.values() if ".py" in t["bash_command"]]
    assert len(script_tasks) == 4
    for t in script_tasks:
        assert t["append_env"] is True
        assert t["env"]["TMP_DIR"] == RUN_TMP


def test_cleanup_removes_only_this_run(monkeypatch):
    _, tasks = _load_dag(monkeypatch)
    cleanup = tasks["cleanup_tmp"]
    assert cleanup["bash_command"] == f"rm -rf {RUN_TMP}"
    assert cleanup["trigger_rule"] == "all_done"


def test_no_timeout_wrapper(monkeypatch):
    _, tasks = _load_dag(monkeypatch)
    assert all("timeout " not in t["bash_command"] for t in tasks.values())
    assert all("execution_timeout" in t for t in tasks.values() if ".py" in t["bash_command"])


def test_run_tmp_does_not_need_a_logical_date():
    # ts_nodash is undefined for a manual run without logical_date in Airflow 3
    assert "ts_nodash" not in RUN_TMP and "run_id" in RUN_TMP
