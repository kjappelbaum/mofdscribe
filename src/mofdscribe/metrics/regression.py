# -*- coding: utf-8 -*-
"""Metrics for the regression setting."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def top_n_in_top_k(
    predictions: ArrayLike, labels: ArrayLike, k: int, n: int, maximize: bool = True
) -> int:
    """Find how many of the top n predictions are in the top k labels.

    Args:
        predictions (ArrayLike): predictions for one objective
        labels (ArrayLike): true labels for one objective
        k (int): number of top labels to consider
        n (int): number of top predictions to consider
        maximize (bool): Set to `True` if larger is better.
            Defaults to True.

    Examples:
        >>> predictions = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> labels = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> top_n_in_top_k(predictions, labels, k=2, n=2)
        2

    Returns:
        int: number of top n predictions in top k labels
    """
    indices_predictions = np.argsort(predictions)
    indices_labels = np.argsort(labels)

    if maximize:
        indices_predictions = indices_predictions[::-1]
        indices_labels = indices_labels[::-1]

    top_n_predictions = indices_predictions[:n]
    top_k_labels = indices_labels[:k]

    return np.sum(np.isin(top_n_predictions, top_k_labels))
