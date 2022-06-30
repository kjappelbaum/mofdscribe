import time
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from mofdscribe.metrics.metric_collection import RegressionMetricCollection

# If i just take an object with those train and fit functions I, in principle, should not care
# if it spawns extra process on GPU or something like that. I just consume its outputs
# In this way, the underlying implementation might also cache the features, for instance, in some file
# It will be pretty hard for us to monitor the resources the run needed as the model function could,
# in principle, spawn extra processess on GPU or something like that.


class BenchResult(BaseModel):
    """Model for benchmark results."""

    time_taken: float
    metrics: RegressionMetricCollection


class MOFBench(ABC):
    """Base class for MOFBench.

    MOFBench is a class that encapsulates the benchmarking of a model.
    It is meant to be used as a base class for other classes that implement
    task specific benchmarking.

    Each task-specific benchmarking class will be specific to a dataset
    and a splitting technique. We will not define feature vectors in the
    benchmarking classes but only indices, structures and labels.

    The benchmarking class will then call the train and predict function
    multiple times and then produce a report.
    With this report, users can make a PR to be added to the leaderboard
    (for which a template exists in which they are also asked to provide a link
    to an archived version of their model code).

    Make sure we also benchmark the execution time of the model.
    """

    _settings = None

    def __init__(self, model, version=None):
        self.model = model
        self.start_time = None
        self.end_time = None
        self._fitted = False
        self._ds = None
        self._splitter = None
        self.version = version

    def _train(self, idx: np.ndarray, structures: np.ndarray, y: np.ndarray):
        self.model.train(idx, structures, y)
        self._fitted = True

    def _predict(self, idx: np.ndarray, structures: np.ndarray):
        return self.model.predict(idx, structures)

    def _score(self):
        raise NotImplementedError

    @property
    def report(self):
        if not self._fitted:
            raise ValueError("Model not fitted yet")
        ...
