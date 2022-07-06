# -*- coding: utf-8 -*-
"""Bases for MOF ML model benchmarking."""
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel

from mofdscribe.datasets.dataset import StructureDataset
from mofdscribe.metrics.metric_collection import RegressionMetricCollection
from mofdscribe.metrics.regression import get_regression_metrics
from mofdscribe.splitters.splitters import Splitter, StratifiedSplitter
from mofdscribe.version import get_version

__all__ = ["MOFBenchRegression", "BenchResult"]


class BenchResult(BaseModel):
    """Model for benchmark results."""

    start_time: datetime
    end_time: datetime
    # ToDo: Add ClassificationMetricCollection
    metrics: RegressionMetricCollection
    version: Optional[str]
    features: Optional[str]
    model_name: Optional[str]
    reference: Optional[str]
    implementation: Optional[str]
    mofdscribe_version: str

    def save_json(self, path: str) -> None:
        """Save benchmark results to json file."""
        with open(path, "w") as handle:
            handle.write(self.json())


class MOFBench(ABC):
    """Base class for MOFBench.

    MOFBench is a class that encapsulates the benchmarking of a model.
    It is meant to be used as a base class for other classes that implement
    task specific benchmarking.

    Each task-specific benchmarking class will be specific to a dataset
    and a splitting technique. We will not define feature vectors in the
    benchmarking classes but only indices, structures and labels.

    The benchmarking class will then call the train and predict function of a model
    multiple times and then produce a report.
    With this report, users can make a PR to be added to the leaderboard
    (for which a template exists in which they are also asked to provide a link
    to an archived version of their model code).

    The model class can be as simple as in the following example

    .. code-block:: python

        class MyDummyModel:
            def __init__(self):
                self.model = Pipeline(
                    [("scaler", StandardScaler()), ("lr", LinearRegression(**(lr_kwargs or {})))]
                )

            def featurize(self, s: Structure):
                return s.density

            def train(self, idx, structures, y):
                x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
                self.model.fit(x, y)

            def predict(self, idx, structures):
                x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
                return self.model.predict(x)

    Of course, instead of re-featurizing every structure you might also
    choose to use the indices to look up pre-computed features.
    """

    def __init__(
        self,
        model,
        ds: StructureDataset,
        splitter: Union[Splitter, StratifiedSplitter],
        target: List[str],
        k: int = 5,
        version: Optional[str] = None,
        features: Optional[str] = None,
        model_name: Optional[str] = None,
        reference: Optional[str] = None,
        implementation: Optional[str] = None,
        debug: bool = False,
    ):
        self._model = model
        self._start_time = None
        self._end_time = None
        self._fitted = False
        self._ds = ds
        self._splitter = splitter
        self._version = version
        self._features = features
        self._model_name = model_name
        self._reference = reference
        self._implementation = implementation
        self._debug = debug
        self._targets = target
        self._k = k

    def _train(self, idx: np.ndarray, structures: np.ndarray, y: np.ndarray):
        self._model.train(idx, structures, y)
        self._fitted = True

    def _predict(self, idx: np.ndarray, structures: np.ndarray):
        return self._model.predict(idx, structures)

    @abstractmethod
    def _score(self):
        raise NotImplementedError

    def bench(self) -> BenchResult:
        start_time = time.time()
        metrics = self._score()
        end_time = time.time()
        return BenchResult(
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            version=self._version,
            features=self._features,
            model_name=self._model_name,
            reference=self._reference,
            implementation=self._implementation,
            mofdscribe_version=get_version(),
        )


class MOFBenchRegression(MOFBench):
    def _score(self):
        metric_collection = []
        timings = []
        inference_times = []
        sample_frac = 0.01 if self._debug else 1.0
        for train_idx, test_idx in self._splitter.k_fold(
            self._ds, self._k, sample_frac=sample_frac
        ):

            start_time = time.time()
            self._train(
                train_idx,
                self._ds.get_structures(train_idx),
                self._ds._df[self._targets].iloc[train_idx],
            )
            end_time = time.time()
            timings.append(end_time - start_time)
            start_time = time.time()
            y_pred = self._predict(test_idx, self._ds.get_structures(test_idx))
            end_time = time.time()
            inference_times.append(end_time - start_time)
            y_true = self._ds._df[self._targets].iloc[test_idx]
            metrics = get_regression_metrics(y_pred, y_true)
            metric_collection.append(metrics)
        return RegressionMetricCollection(
            regression_metrics=metric_collection,
            fit_timings=timings,
            inference_timings=inference_times,
        )
