# -*- coding: utf-8 -*-
"""Test the functions in the regression metrics module."""
import numpy as np

from mofdscribe.metrics.regression import RegressionMetrics, get_regression_metrics, top_n_in_top_k


def test_top_n_in_top_k():
    """Test the identity of indiscernibles property for top_n_in_top_k."""
    true = np.arange(10)
    pred = np.arange(10)

    assert top_n_in_top_k(pred, true, k=1, n=1) == 1
    assert top_n_in_top_k(pred, true, k=1, n=2) == 1
    assert top_n_in_top_k(pred, true, k=2, n=2) == 2

    pred = pred[::-1]
    assert top_n_in_top_k(pred, true, k=1, n=1) == 0


def test_get_regression_metrics():
    """Test call to get_regression_metrics.

    * Test dentity of indiscernibles property
    * Test return type is RegressionMetrics
    """
    true = np.arange(10)
    pred = np.arange(10)

    res = get_regression_metrics(pred, true)
    assert isinstance(res, RegressionMetrics)
    for k, v in res.dict().items():
        if "top" not in k:
            assert isinstance(v, float)
        else:
            assert isinstance(v, int)
        if k == "r2_score":
            assert v == 1.0
        elif "top" in k:
            assert v >= 1
        else:
            assert v == 0
