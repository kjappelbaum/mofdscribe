# -*- coding: utf-8 -*-
import numpy as np


def top_n_in_top_k(
    predictions: np.array, labels: np.array, k: int, n: int, maximize: bool = True
) -> int:
    """Find how many of the top n predictions are in the top k labels.

    Args:
        predictions (np.array): predictions for one objective
        labels (np.array): true labels for one objective
        k (int): number of top labels to consider
        n (int): number of top predictions to consider
        maximize (bool, optional): Set to `True` if larger is better.
            Defaults to True.

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


def gdofaic(model, X, y, epsilon=1e-5, mc_samples=10) -> float:
    """Computes the Akaike Information Criterion (AIC) for a model using the generalized degrees of freedom."""
    ...
