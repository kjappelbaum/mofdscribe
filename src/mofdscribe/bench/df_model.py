# -*- coding: utf-8 -*-
"""Helper to build bench model for models that operate on pre-computed feature frames."""

from typing import Iterable, Optional

from loguru import logger

__all__ = ["DFModel"]


class DFModel:
    def __init__(self, model, features: Optional[Iterable[str]] = None):
        """Initialize the model.

        .. note:::

            If you use the model, you must set
            :code:`patch_in_ds` in the
            :code:py:`~mofdscribe.bench.mofbench.MOFBench` class.

        Args:
            model (object): Must implement `fit` and `predict` methods.
                Using a sklearn function signature will work.
            features (Iterable[str], optional): Feature names to use.
                If not provided, all features will be used.
                Defaults to None.
        """
        self._model = model
        self._features = features

    @property
    def features(self):
        if self._features is None:
            self._features = list(self.ds.available_features)
        return self._features

    def fit(self, idx, structures, y):
        logger.debug("Fitting model")
        X = self.ds._df[self.features].loc[idx, :]  # noqa: N806
        logger.debug(X.shape)
        self._model.fit(X, y)  # noqa: N806

    def predict(self, idx, structures):
        X = self.ds._df[self.features].loc[idx, :]  # noqa: N806
        return self._model.predict(X)  # noqa: N806
