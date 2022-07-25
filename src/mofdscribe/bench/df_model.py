"""Helper to build bench model for models that operate on pre-computed feature frames."""
import pandas as pd


class DFModel:
    def __init__(self, model, feature_df: pd.DataFrame):
        """Initialize the model.

        Args:
            model (object): Must implement `fit` and `predict` methods.
                Using a sklearn function signature will work.
            feature_df (pd.DataFrame): Feature dataframe.
        """
        self._model = model
        self._feature_df = feature_df

    def fit(self, idx, structures, y):
        X = self._feature_df.loc[idx, :]  # noqa: N806
        self._model.fit(X, y)  # noqa: N806

    def predict(self, idx, structures):
        X = self._feature_df.loc[idx, :]  # noqa: N806
        return self._model.predict(X)  # noqa: N806
