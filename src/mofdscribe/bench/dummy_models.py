# -*- coding: utf-8 -*-
"""Simple baseline models."""
from typing import Dict, Optional

import numpy as np
from pymatgen.core import Structure
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

__all__ = ["DensityRegressor"]


class DensityRegressor:
    """Dummy model."""

    def __init__(self, lr_kwargs: Optional[Dict] = None):
        """Initialize the model.

        Args:
            lr_kwargs (Optional[Dict], optional): Keyword arguments
                that are passed to the linear regressor.
                Defaults to None.
        """
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=3)),
                ("lr", LinearRegression(**(lr_kwargs or {}))),
            ]
        )

    def featurize(self, s: Structure) -> float:
        """You might want to use a lookup in some dataframe instead.

        Or use some mofdscribe featurizers.

        Args:
            s (Structure): Structure to featurize.

        Returns:
            float: Density of the structure.
        """
        return s.density

    def fit(self, idx, structures, y):
        x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
        self.model.fit(x, y)

    def predict(self, idx, structures):
        x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
        return self.model.predict(x)


class DummyRegressor:
    """Dummy regressor model."""

    def __init__(self, strategy: str = "mean"):
        """Initialize the model.

        Args:
            strategy (str): Strategy to use for prediction.
                Defaults to "mean".
        """
        self.model = SklearnDummyRegressor(strategy=strategy)

    def fit(self, idx, structures, y):
        self.model.fit(idx, y)

    def predict(self, idx, structures):
        return self.model.predict(idx)
