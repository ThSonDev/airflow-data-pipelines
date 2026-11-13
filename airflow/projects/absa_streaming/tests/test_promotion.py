import json

import pytest
import save_postgres
import torch
from absa_core import ASPECTS, EMBED_DIM, TextCNN


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    return tmp_path


class FakeConn:
    def __init__(self, fail_insert=False):
        self.fail_insert, self.committed, self.rolled_back, self.closed = fail_insert, False, False, False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        if self.fail_insert and sql.lstrip().startswith("INSERT"):
            raise RuntimeError("insert failed")

    def fetchone(self):
        return (7,)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


@pytest.fixture
def tmp_run(models_dir, monkeypatch):
    run = models_dir / "tmp"
    run.mkdir()
    torch.save(TextCNN(10, EMBED_DIM, len(ASPECTS)).state_dict(), run / "model_temp.pth")
    (run / "vocab.json").write_text("{}")
    (run / "config.json").write_text(json.dumps({"mask_padding": True}))
    metrics = {"candidate": {"macro_f1": 0.7, "balanced_acc": 0.6}, "production": {"name": "cnn_best", "macro_f1": 0.5}}
    (run / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setenv("TMP_DIR", str(run))
    return models_dir


def test_promotion_creates_folder(tmp_run, monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(save_postgres, "connect", lambda: conn)
    save_postgres.main()
    folder = tmp_run / "newer_models" / "cnn_v7"
    assert sorted(p.name for p in folder.iterdir()) == ["config.json", "metrics.json", "model.pth", "vocab.json"]
    assert conn.committed and conn.closed
    assert not (tmp_run / "newer_models" / "cnn_v7.tmp").exists()


def test_insert_failure_removes_folder_and_exits(tmp_run, monkeypatch):
    conn = FakeConn(fail_insert=True)
    monkeypatch.setattr(save_postgres, "connect", lambda: conn)
    with pytest.raises(SystemExit) as exc:
        save_postgres.main()
    assert exc.value.code == 1
    assert conn.rolled_back and not conn.committed
    assert not (tmp_run / "newer_models" / "cnn_v7").exists()


def test_connect_failure_exits_nonzero(tmp_run, monkeypatch):
    def refuse():
        raise OSError("no db")

    monkeypatch.setattr(save_postgres, "connect", refuse)
    with pytest.raises(SystemExit) as exc:
        save_postgres.main()
    assert exc.value.code == 1


def test_worse_candidate_is_not_promoted(tmp_run, monkeypatch):
    metrics = json.loads((tmp_run / "tmp" / "metrics.json").read_text())
    metrics["production"]["macro_f1"] = 0.9
    (tmp_run / "tmp" / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(save_postgres, "connect", lambda: pytest.fail("DB must not be touched"))
    save_postgres.main()
    assert not (tmp_run / "newer_models").exists()
