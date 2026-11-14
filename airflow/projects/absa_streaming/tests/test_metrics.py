import numpy as np
import pytest
from absa_core import ASPECTS
from metrics import compute_metrics


def test_perfect_predictions():
    y = np.tile(np.arange(4)[:, None], (1, 8))  # every aspect column holds all 4 classes
    m = compute_metrics(y, y)
    assert m["mean_aspect_f1"] == pytest.approx(1.0)
    assert m["macro_f1_flat"] == pytest.approx(1.0)
    assert m["balanced_acc"] == pytest.approx(1.0)


def test_known_values_and_none_only_aspect():
    # aspect 0: truth [0, 1, 1, 1], predicted [0, 0, 1, 1]
    # class 0: P=1/2 R=1 F1=2/3; class 1: P=1 R=2/3 F1=0.8; mean = (2/3 + 0.8) / 2
    # aspect 1: only NONE in truth and prediction -> one class counted, F1 = 1.0
    y_true = np.zeros((4, 8), dtype=int)
    y_pred = np.zeros((4, 8), dtype=int)
    y_true[:, 0] = [0, 1, 1, 1]
    y_pred[:, 0] = [0, 0, 1, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["aspect_f1"][ASPECTS[0]] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["aspect_f1"][ASPECTS[1]] == pytest.approx(1.0)
    assert m["mean_aspect_f1"] == pytest.approx(np.mean(list(m["aspect_f1"].values())))
    assert set(m["aspect_f1"]) == set(ASPECTS)


def test_predicting_a_class_absent_from_truth_is_penalized():
    # truth all NONE (8 rows), 2 rows predicted NEG: class 0 F1 = 6/7, class 1 F1 = 0 -> mean 3/7
    y_true = np.zeros((8, 8), dtype=int)
    y_pred = y_true.copy()
    y_pred[:2, 0] = 1
    assert compute_metrics(y_true, y_pred)["aspect_f1"][ASPECTS[0]] == pytest.approx(3 / 7)
