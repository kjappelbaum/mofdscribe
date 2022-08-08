# -*- coding: utf-8 -*-
"""Helper to build bench model for models that operate on pre-computed feature frames."""

from typing import Iterable


class DFModel:
    def __init__(self, model, features: Iterable[str]):
        """Initialize the model.

        .. note:::

            If you use the model, you must set
            :code:`patch_in_ds` in the
            :code:py:`~mofdscribe.bench.mofbench.MOFBench` class.

        Args:
            model (object): Must implement `fit` and `predict` methods.
                Using a sklearn function signature will work.
            feature_df (pd.DataFrame): Feature dataframe.
        """
        self._model = model
        self._features = features

    def fit(self, idx, structures, y):
        X = self.ds._df[self._features].loc[idx, :]  # noqa: N806
        self._model.fit(X, y)  # noqa: N806

    def predict(self, idx, structures):
        X = self.ds._df[self._features].loc[idx, :]  # noqa: N806
        return self._model.predict(X)  # noqa: N806
