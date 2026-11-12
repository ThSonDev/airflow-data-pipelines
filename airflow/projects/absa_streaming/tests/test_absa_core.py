import json
from pathlib import Path

import pytest
import torch
from absa_core import (
    ASPECTS,
    EMBED_DIM,
    NUM_CLASSES,
    TextCNN,
    clean,
    encode,
    get_settings,
    remove_vietnamese_tone,
)

MODELS = Path(__file__).resolve().parents[3] / "models"


def test_clean_lowercases_and_strips():
    assert clean("  Giao HÀNG nhanh  ") == "giao hàng nhanh"


def test_clean_removes_noise_and_collapses_spaces():
    assert clean("đẹp!!! quá :)   luôn") == "đẹp quá luôn"


def test_clean_keeps_digits_and_accents():
    assert clean("Giá 99k") == "giá 99k"


def test_clean_non_string():
    assert clean(None) == "none"


def test_encode_unknown_maps_to_1():
    vocab = {"<PAD>": 0, "<UNK>": 1, "đẹp": 2}
    assert encode("đẹp xấu", vocab) == [2, 1]


def test_model_output_shape():
    model = TextCNN(50, EMBED_DIM, len(ASPECTS)).eval()
    x = torch.randint(2, 50, (3, 12))
    assert model(x).shape == (3, len(ASPECTS), NUM_CLASSES)


def test_get_settings_env_override(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("MODELS_DIR", "/m")
    s = get_settings()
    assert s.postgres_host == "localhost"
    assert s.tmp_dir == "/m/tmp"


def test_get_settings_defaults(monkeypatch):
    for k in ("POSTGRES_HOST", "MODELS_DIR", "TMP_DIR", "KAFKA_BOOTSTRAP"):
        monkeypatch.delenv(k, raising=False)
    s = get_settings()
    assert (s.postgres_host, s.kafka_bootstrap) == ("postgres", "kafka:9092")
    assert s.tmp_dir == "/opt/airflow/models/tmp"


@pytest.mark.skipif(not (MODELS / "cnn_best.pth").exists(), reason="model file missing")
def test_cnn_best_loads_strict():
    vocab = json.loads((MODELS / "vocab.json").read_text(encoding="utf-8"))
    model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS))
    model.load_state_dict(torch.load(MODELS / "cnn_best.pth", map_location="cpu"), strict=True)


def test_remove_vietnamese_tone():
    assert remove_vietnamese_tone("đẹp quá") == "đep qua"
