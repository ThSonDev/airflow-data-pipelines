"""Model loading and the pandas UDF used by the Spark consumer."""
import pandas as pd
import torch
from absa_core import PREDICTION_MAP, get_logger
from model_registry import load_model, predict_classes, resolve_production_model
from pyspark.sql import types as T
from pyspark.sql.functions import pandas_udf

log = get_logger("inference")

# Loaded once per Python worker
_model = None
_vocab = None
_device = None


def load_resources():
    global _model, _vocab, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        info = resolve_production_model()
        _model, _vocab = load_model(info, _device)
        log.info("Model %s loaded from %s on %s", info["name"], info["model_path"], _device)


def _predict(texts: pd.Series) -> pd.Series:
    load_resources()
    if len(texts) == 0:
        return pd.Series([[] for _ in texts])
    predictions = predict_classes(_model, _vocab, texts, _device)  # (batch, 8)
    return pd.Series([[PREDICTION_MAP.get(p, "NONE") for p in row] for row in predictions])


# Wrap after defining: a decorator would hide _predict, and cloudpickle would then copy the
# function with its own globals (model stays None) instead of pickling it by reference.
cnn_inference_udf = pandas_udf(_predict, T.ArrayType(T.StringType()))
