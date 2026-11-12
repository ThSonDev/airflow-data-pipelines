import os
import subprocess
import sys
from pathlib import Path

import inference
import pytest
from pyspark import cloudpickle

ABSA_DIR = Path(__file__).resolve().parents[1]
MODELS = ABSA_DIR.parents[1] / "models"

LOAD_AND_RUN = """
import pickle, sys
import pandas as pd
f = pickle.load(open(sys.argv[1], "rb"))
out = f(pd.Series(["giao hàng nhanh", "tệ"]))
assert len(out) == 2 and all(len(row) == 8 for row in out), out
print("ok")
"""


@pytest.mark.skipif(not (MODELS / "cnn_best.pth").exists(), reason="model file missing")
def test_udf_works_after_pickle_in_fresh_process(tmp_path):
    # Spark pickles the UDF and runs it in a new Python worker. The model globals must live in
    # the inference module, not in a by-value copy of the function.
    path = tmp_path / "udf.pkl"
    path.write_bytes(cloudpickle.dumps(inference.cnn_inference_udf.func))
    env = {
        **os.environ,
        "PYTHONPATH": str(ABSA_DIR / "scripts"),
        "MODELS_DIR": str(MODELS),
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "1",  # unused port: the DB lookup fails fast and cnn_best.pth is used
    }
    result = subprocess.run(
        [sys.executable, "-c", LOAD_AND_RUN, str(path)], env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "ok" in result.stdout
