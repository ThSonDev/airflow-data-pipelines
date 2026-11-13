import json
from pathlib import Path

import pytest
import torch
from model_registry import load_model, resolve_production_model, should_promote

REAL_MODELS = Path(__file__).resolve().parents[3] / "models"


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    return tmp_path


def _folder(models_dir, name, files=("model.pth", "vocab.json")):
    folder = models_dir / "newer_models" / name
    folder.mkdir(parents=True)
    for f in files:
        (folder / f).write_text("x")
    return folder


@pytest.mark.parametrize("new,prod,expected", [(0.7, 0.6, True), (0.6, 0.6, False), (0.5, 0.6, False)])
def test_should_promote(new, prod, expected):
    assert should_promote(new, prod) is expected


def test_valid_folder_is_used(models_dir):
    folder = _folder(models_dir, "cnn_v3")
    (folder / "config.json").write_text(json.dumps({"mask_padding": True}))
    info = resolve_production_model(lambda: "cnn_v3")
    assert info["name"] == "cnn_v3"
    assert info["model_path"] == str(folder / "model.pth")
    assert info["vocab_path"] == str(folder / "vocab.json")
    assert info["config"]["mask_padding"] is True


@pytest.mark.parametrize("name", ["../x", "/abs", "cnn_v1.pth", "cnn_v1/../cnn_v2", "cnn_v", None])
def test_bad_names_use_legacy(models_dir, name):
    _folder(models_dir, "cnn_v1")
    info = resolve_production_model(lambda: name)
    assert info["name"] == "cnn_best"
    assert info["model_path"] == str(models_dir / "cnn_best.pth")
    assert info["vocab_path"] == str(models_dir / "vocab.json")


def test_folder_missing_vocab_uses_legacy(models_dir):
    _folder(models_dir, "cnn_v2", files=("model.pth",))
    assert resolve_production_model(lambda: "cnn_v2")["name"] == "cnn_best"


def test_fetch_error_uses_legacy(models_dir):
    def boom():
        raise RuntimeError("db down")

    assert resolve_production_model(boom)["name"] == "cnn_best"


@pytest.mark.skipif(not (REAL_MODELS / "cnn_best.pth").exists(), reason="model file missing")
def test_load_legacy_model_strict(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(REAL_MODELS))
    info = resolve_production_model(lambda: None)
    model, vocab = load_model(info, torch.device("cpu"))
    assert model.embedding.num_embeddings == len(vocab)
    assert model.mask_padding is False and not model.training
