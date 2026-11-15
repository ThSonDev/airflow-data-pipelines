from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sink


def _cfg(tmp_path, crash_batch=-1):
    return SimpleNamespace(
        postgres_host="h", postgres_port=1, postgres_db="d", postgres_user="u", postgres_password="p",
        checkpoint_dir=str(tmp_path), crash_batch=crash_batch,
    )


def _df(rows=0):
    df = MagicMock()
    df.count.return_value = rows
    return df


def test_write_error_is_re_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(sink, "cfg", _cfg(tmp_path))
    df = _df()
    w = MagicMock()
    w.format.return_value = w.option.return_value = w.mode.return_value = w
    w.save.side_effect = RuntimeError("db down")
    df.select.return_value.write = w
    with pytest.raises(RuntimeError, match="db down"):
        sink.write_to_postgres(df, 1)


def test_crash_off_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(sink, "cfg", _cfg(tmp_path))
    sink.write_to_postgres(_df(), 7)


def test_simulated_crash_happens_once(monkeypatch, tmp_path):
    monkeypatch.setattr(sink, "cfg", _cfg(tmp_path, crash_batch=7))
    with pytest.raises(RuntimeError, match="Simulated crash"):
        sink.write_to_postgres(_df(), 7)
    assert (tmp_path / "crash_done").exists()
    sink.write_to_postgres(_df(), 7)
