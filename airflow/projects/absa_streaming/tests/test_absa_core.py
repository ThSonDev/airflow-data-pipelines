import json
import unicodedata
from pathlib import Path

import pytest
import torch
from absa_core import (
    ASPECTS,
    EMBED_DIM,
    NUM_CLASSES,
    WORD_WINDOW,
    TextCNN,
    clean,
    encode,
    get_settings,
    pad_batch,
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


def test_clean_emoji_and_punctuation_only():
    assert clean("😍😍😍").split() == []
    assert clean("!!!").split() == []
    assert clean("") == ""


def test_clean_nfd_equals_nfc():
    nfc = "đẹp quá"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc
    assert clean(nfd) == clean(nfc) == nfc


@pytest.mark.parametrize("text", ["😍😍😍", "!!!", ""])
def test_encode_empty_gives_unk(text):
    assert encode(clean(text), {"<PAD>": 0, "<UNK>": 1}) == [1]


def test_pad_batch_min_width_and_padding():
    out = pad_batch([[2, 3], [4]])
    assert out.shape == (2, WORD_WINDOW)
    assert out.tolist() == [[2, 3, 0, 0, 0], [4, 0, 0, 0, 0]]
    assert pad_batch([list(range(2, 10))]).shape == (1, 8)


def test_all_empty_batch_runs_through_model():
    vocab = {"<PAD>": 0, "<UNK>": 1}
    batch = pad_batch([encode(clean(t), vocab) for t in ["😍😍😍", "!!!", ""]])
    model = TextCNN(10, EMBED_DIM, len(ASPECTS)).eval()
    assert model(batch).shape == (3, len(ASPECTS), NUM_CLASSES)
