# -*- coding: utf-8 -*-
"""Helpers for adverserial validation"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

__all__ = ["AdverserialValidator"]


class AdverserialValidator:
    """Helper for adverserial validation.

    Adverserial is a method to estimate how different two datasets are.
    Most commonly, it is used to estimate if the train and test sets
    come from the same distribution.
    It has found widespread use in data science competions [KaggleBook]_,
    but more recently, also in some auto-ML systems.

    The basic idea is quite simple: Train a classifier to distinguish
    two datasets. If it can learn to do so, there are differences,
    if it cannot, then the datasets are indistinguishable.
    Additionally, this approach also enables us to investigate
    what the most important features for this difference are.
    If one aims to reduce data drift problems, one might remove those
    features [Uber]_.

    Here, we use simple ensemble classifiers such as random forests
    or extra trees to compute the k-fold crossvalidated area under
    the receiver-operating characteristic curve.

    Example:
        >>> from mofdscribe.metrics.adverserial import AdverserialValidator
        >>> x_a = np.array([[1, 2, 3], [4, 5, 6]])
        >>> x_b = np.array([[1, 2, 3], [4, 5, 6]])
        >>> validator = AdverserialValidator(x_a, x_b)
        >>> validator.score().mean()
        0.5

    References:
        .. [Uber] Pan, J.; Pham, V.; Dorairaj, M.; Chen, H.; Lee, J.-Y.
            arXiv:2004.03045 June 26, 2020.

        .. [KaggleBook] Banachewicz, K.; Massaron, L.
            The Kaggle Book: Data Analysis and Machine Learning
            for Competitive Data Science; Packt Publishing, 2022.
    """

    def __init__(
        self,
        x_a: Union[ArrayLike, pd.DataFrame],
        x_b: Union[ArrayLike, pd.DataFrame],
        modeltype: str = "rf",
        k: int = 5,
    ):
        """Initiate a AdverserialValidator instance.

        Args:
            x_a (Union[ArrayLike, pd.DataFrame]): Data for the first dataset (e.g. training set).
            x_b (Union[ArrayLike, pd.DataFrame]): Data for the second dataset (e.g. test set).
            modeltype (str): Classifier to train. Defaults to "rf".
            k (int): Number of folds in k-fold crossvalidation. Defaults to 5.

        Raises:
            ValueError: If the chosen modeltype is not supported.
        """
        if modeltype == "rf":
            self.model = RandomForestClassifier()
        elif modeltype == "et":
            self.model = ExtraTreesClassifier()
        else:
            raise ValueError(f"Model {modeltype} not implements. Available models are rf, et.")

        self.x_a = x_a
        self.x_b = x_b
        self.k = k

    def _get_x_y(self):
        if isinstance(self.x_a, pd.DataFrame):
            x = pd.concat([self.x_a, self.x_b])

        else:
            x = np.vstack([self.x_a, self.x_b])

        y = [0] * len(self.x_a) + [1] * len(self.x_b)

        return x, y

    def score(self) -> np.array:
        """Compute the area under the receiver-operating characteristic curve.

        A score close to 0.5 means that the two datasets are similar.

        Returns:
            np.array: Areas under the receiver-operating characteristic curve.

        """
        x, y = self._get_x_y()
        score = cross_val_score(self.model, x, y, scoring="roc_auc")
        return score

    def get_feature_importance(self) -> np.array:
        """Identify the features distinguishing the two datasets.

        Uses the default impurity-based feature importance.

        Returns:
            np.array: Feature importance scores.
        """
        x, y = self._get_x_y()
        self.model.fit(x, y)

        importances = self.model.feature_importances_
        return importances
