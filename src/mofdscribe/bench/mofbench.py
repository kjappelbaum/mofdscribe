# -*- coding: utf-8 -*-
"""Bases for MOF ML model benchmarking."""
import base64
import hashlib
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from mofdscribe.bench.watermark import get_watermark
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.metrics.metric_collection import RegressionMetricCollection
from mofdscribe.metrics.regression import get_regression_metrics
from mofdscribe.splitters.splitters import BaseSplitter
from mofdscribe.version import get_version

__all__ = ["MOFBenchRegression", "BenchResult"]

_RST_TEMPLATE = """{modelname}
------------------------------------

Model card
..............

Feature set description
~~~~~~~~~~~~~~~~~~~~~~~~~~

What features are used?
#######################

Why are the features informative?
###################################


Why do the features not leak information?
##############################################


Will the features be accessible in real-world test applications?
###################################################################

Data split
~~~~~~~~~~

Describe preprocessing steps and how data leakage is avoided
##############################################################

Describe the feature selection steps and how data leakage is avoided
#####################################################################


Describe the model selection steps and how data leakage is avoided
#####################################################################

"""

_CITATION_TEMPLATE = """"
Reference
..............

.. code::

    {bibtex}
"""


class BenchTaskEnum(Enum):
    """Enum for benchmark tasks."""

    logKH_CO2_id = "logKH_CO2_id"  # noqa: N815
    logKH_CO2_ood = "logKH_CO2_ood"  # noqa: N815
    pbe_bandgap_id = "pbe_bandgap_id"  # noqa: N815
    ch4dc_id = "ch4dc_id"  # noqa: N815


class BenchResult(BaseModel):
    """Model for benchmark results."""

    start_time: datetime = Field(title="Start time", description="Start time of the benchmark")
    end_time: datetime = Field(title="End time", description="End time of the benchmark")
    # ToDo: Add ClassificationMetricCollection
    metrics: RegressionMetricCollection = Field(
        title="Metrics", description="Metrics of the benchmark"
    )
    version: Optional[str] = Field(title="Version", description="Version of the benchmark")
    features: Optional[str] = Field(
        title="Features",
        description="Features used in the model. If you use a graph net, report the edge and vertex features.",
    )
    name: str = Field(title="Name", description="Name of the model. This will be used as filename.")
    task: BenchTaskEnum = Field(title="Task", description="Task of the benchmark")
    model_type: Optional[str] = Field(
        title="Model type", description="Model type, e.g. 'CGCNN', 'BERT', 'XGBoost'"
    )
    reference: Optional[str] = Field(title="Reference", description="Reference with more details")
    implementation: Optional[str] = Field(
        title="Implementation", description="Link to implementation"
    )
    mofdscribe_version: str = Field(title="mofdscribe version", description="mofdscribe version")
    session_info: Dict[str, Any] = Field(
        title="Session info",
        description="Automatically captured string describing the computational environment.",
    )
    logs: Optional[List[Dict]] = Field(
        title="Logs",
        description="Additional data logged using the log method during the benchmark.",
    )

    def save_json(self, folder: Union[str, os.PathLike]) -> None:
        """Save benchmark results to json file."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(
            os.path.join(
                folder,
                f"{id_for_bench_result(self)}.json",
            ),
            "w",
        ) as handle:
            handle.write(self.json())

    def save_rst(self, folder: Union[str, os.PathLike]) -> None:
        """Prepare RST file for model card and free-text model description."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(
            os.path.join(
                folder,
                f"{id_for_bench_result(self)}.rst",
            ),
            "w",
        ) as handle:
            text = _RST_TEMPLATE.format(modelname=self.name)
            if self.reference is not None:
                text += _CITATION_TEMPLATE.format(bibtex=self.reference)
            handle.write(text)


def id_for_bench_result(bench_result: BenchResult) -> str:
    """Generate an ID for a benchmark result."""
    hasher = hashlib.sha1(str(bench_result).encode("utf8"))  # noqa: S303 ok to use unsafe hash here
    hash = base64.urlsafe_b64encode(hasher.digest()).decode("ascii")

    n = bench_result.name[:2] if bench_result.name else hash[:2]
    f = bench_result.features[:2] if bench_result.features else hash[2:4]
    m = bench_result.model_type[:2] if bench_result.model_type else hash[4:6]
    r = bench_result.reference[:2] if bench_result.reference else hash[6:8]
    datetime_part = bench_result.start_time.strftime("%Y%m%d%H%M%S")
    return f"R{n}{f}{m}{r}{datetime_part}".replace("/", "-").replace(" ", "_")


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

            def fit(self, idx, structures, y):
                x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
                self.model.fit(x, y)

            def predict(self, idx, structures):
                x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
                return self.model.predict(x)

    Of course, instead of re-featurizing every structure you might also
    choose to use the indices to look up pre-computed features.

    ..warning::

        This class will monkey path a :code:`log` method into the model object.
    """

    def __init__(
        self,
        model,
        ds: AbstractStructureDataset,
        splitter: BaseSplitter,
        target: List[str],
        name: str,
        task: BenchTaskEnum,
        k: int = 5,
        model_type: Optional[str] = None,
        version: Optional[str] = None,
        features: Optional[str] = None,
        reference: Optional[str] = None,
        implementation: Optional[str] = None,
        debug: bool = False,
        patch_in_ds: bool = False,
    ):
        """Initialize the benchmarking class.

        Args:
            model: Model to be benchmarked.
            ds (AbstractStructureDataset): Dataset to be used for benchmarking.
            splitter (BaseSplitter): Splitter to be used for benchmarking.
            target (List[str]): Target labels to be used for benchmarking.
                Must be included in the dataset.
            name (str): Name of the model. This will be used as filename.
            task (BenchTaskEnum): Task of the benchmark.
            k (int): Number of folds for k-fold cross-validation.
            model_type (str, optional): Model type, e.g. 'CGCNN', 'BERT', 'XGBoost'.
                Defaults to None.
            version (str, optional): Version of the model. Defaults to None.
            features (str, optional): Features used in the model. If you use a graph net,
                report the edge and vertex features. Defaults to None.
            reference (str, optional): Reference with more details
                Please provide it in BibTeX form. Defaults to None.
            implementation (str, optional): Link to implementation. Defaults to None.
            debug (bool): If True, the benchmark will be run in debug mode
                (1% of the data).
            patch_in_ds (bool): If True, the dataset will be patched into the model class
                under the `ds` attribute.
        """
        self._model = model
        self._start_time = None
        self._end_time = None
        self._fitted = False
        self._ds = ds
        self._splitter = splitter
        self._task = task
        self._version = version
        self._features = features
        self._name = name
        self._model_type = model_type
        self._reference = reference
        self._implementation = implementation
        self._debug = debug
        self._targets = target
        self._k = k
        if patch_in_ds:
            self._model.ds = self._ds
        # mokey patch for logging method
        self._model.log = self.log
        self.logs = []

    def _train(self, idx: np.ndarray, structures: np.ndarray, y: np.ndarray):
        self._model.fit(idx, structures, y)
        self._fitted = True

    def _predict(self, idx: np.ndarray, structures: np.ndarray):
        return self._model.predict(idx, structures)

    def log(self, data: dict):
        """Log data to the logs list.

        Args:
            data (dict): Data to be logged.
        """
        self.logs.append(data)

    @abstractmethod
    def _score(self):
        raise NotImplementedError

    def bench(self) -> BenchResult:
        """Run the benchmarking."""
        start_time = time.time()
        metrics = self._score()
        end_time = time.time()
        return BenchResult(
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            version=self._version,
            features=self._features,
            name=self._name,
            model_type=self._model_type,
            task=self._task,
            reference=self._reference,
            implementation=self._implementation,
            mofdscribe_version=get_version(),
            session_info=get_watermark(),
            logs=self.logs,
        )


class MOFBenchRegression(MOFBench):
    """Regression benchmarking class."""

    def _score(self):
        metric_collection = []
        timings = []
        inference_times = []

        for i, (train_idx, test_idx) in enumerate(
            self._splitter.k_fold(
                self._k,
            )
        ):
            logger.debug(
                f"K-fold round {i}, {len(train_idx)} train points, {len(test_idx)} test points"
            )
            start_time = time.time()
            self._train(
                train_idx,
                self._ds.get_structures(train_idx),
                self._ds._df[self._targets].iloc[train_idx].values,
            )
            end_time = time.time()
            timings.append(end_time - start_time)
            start_time = time.time()
            y_pred = self._predict(test_idx, self._ds.get_structures(test_idx))
            end_time = time.time()
            inference_times.append(end_time - start_time)
            y_true = self._ds._df[self._targets].iloc[test_idx].values
            metrics = get_regression_metrics(y_pred, y_true)
            metric_collection.append(metrics)
        return RegressionMetricCollection(
            regression_metrics=metric_collection,
            fit_timings=timings,
            inference_timings=inference_times,
        )
