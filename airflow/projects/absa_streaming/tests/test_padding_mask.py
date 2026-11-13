import json

import pytest
import torch
from absa_core import ASPECTS, EMBED_DIM, TextCNN, load_model_config, pad_batch

SHORT = [5, 9, 12]
LONG = list(range(2, 302))


def _model(mask):
    torch.manual_seed(0)
    return TextCNN(400, EMBED_DIM, len(ASPECTS), mask_padding=mask).eval()


def _logits(model, id_lists):
    with torch.no_grad():
        return model(pad_batch(id_lists))


def test_masked_logits_do_not_depend_on_batch_mates():
    model = _model(True)
    alone = _logits(model, [SHORT])[0]
    batched = _logits(model, [SHORT, LONG])[0]
    assert torch.allclose(alone, batched, atol=1e-6)
    assert torch.isfinite(batched).all()


def test_legacy_model_depends_on_padding():
    model = _model(False)
    alone = _logits(model, [SHORT])[0]
    batched = _logits(model, [SHORT, LONG])[0]
    assert not torch.allclose(alone, batched, atol=1e-6)


def test_missing_config_means_legacy(tmp_path):
    assert load_model_config(str(tmp_path / "model.pth")) == {"mask_padding": False}


def test_config_file_is_read(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"mask_padding": True}))
    assert load_model_config(str(tmp_path / "model.pth"))["mask_padding"] is True


@pytest.mark.parametrize("ids", [[1], [1, 1]])
def test_unk_only_rows_stay_finite(ids):
    assert torch.isfinite(_logits(_model(True), [ids, LONG])).all()
