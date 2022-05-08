import numpy as np

from mofdscribe.metrics.regression import top_n_in_top_k


def test_top_n_in_top_k():
    true = np.arange(10)
    pred = np.arange(10)

    assert top_n_in_top_k(pred, true, k=1, n=1) == 1
    assert top_n_in_top_k(pred, true, k=1, n=2) == 1
    assert top_n_in_top_k(pred, true, k=2, n=2) == 2

    pred = pred[::-1]
    assert top_n_in_top_k(pred, true, k=1, n=1) == 0
