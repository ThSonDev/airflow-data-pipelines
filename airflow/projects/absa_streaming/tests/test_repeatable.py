import pandas as pd
import torch
from absa_core import get_settings, set_seed
from text_utils import drop_overlap


def test_drop_overlap_removes_train_rows_found_in_val_or_test():
    train = pd.DataFrame({"clean_text": ["a", "b", "c", "d"], "x": [1, 2, 3, 4]})
    val = pd.DataFrame({"clean_text": ["b"]})
    test = pd.DataFrame({"clean_text": ["d", "z"]})
    out, dropped = drop_overlap(train, val, test)
    assert dropped == 2
    assert out["clean_text"].tolist() == ["a", "c"]
    assert out["x"].tolist() == [1, 3]


def test_drop_overlap_without_overlap_keeps_everything():
    train = pd.DataFrame({"clean_text": ["a", "b"]})
    out, dropped = drop_overlap(train, pd.DataFrame({"clean_text": ["c"]}))
    assert dropped == 0 and len(out) == 2


def test_set_seed_repeats_random_numbers():
    set_seed(7)
    first = torch.rand(5)
    set_seed(7)
    assert torch.equal(first, torch.rand(5))


def test_seed_setting(monkeypatch):
    monkeypatch.delenv("SEED", raising=False)
    assert get_settings().seed == 42
    monkeypatch.setenv("SEED", "43")
    assert get_settings().seed == 43
