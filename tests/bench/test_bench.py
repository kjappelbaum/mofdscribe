# -*- coding: utf-8 -*-
"""Test the basic benchmark scaffolding."""
import os
from typing import Dict, Optional

import numpy as np
import pytest
from pymatgen.core import Structure
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mofdscribe.bench.mofbench import BenchResult, MOFBench, MOFBenchRegression, id_for_bench_result
from mofdscribe.datasets import CoREDataset
from mofdscribe.splitters import ClusterStratifiedSplitter
from mofdscribe.splitters.splitters import BaseSplitter


@pytest.mark.xdist_group(name="core-ds")
class MyDummyModel:
    """Dummy model."""

    def __init__(self, lr_kwargs: Optional[Dict] = None):
        """Initialize the model.

        Args:
            lr_kwargs (Optional[Dict], optional): Keyword arguments
                that are passed to the linear regressor.
                Defaults to None.
        """
        self.model = Pipeline(
            [("scaler", StandardScaler()), ("lr", LinearRegression(**(lr_kwargs or {})))]
        )

    def featurize(self, s: Structure):
        """You might want to use a lookup in some dataframe instead.

        Or use some mofdscribe featurizers.
        """
        return s.density

    def fit(self, idx, structures, y):
        self.log({"info": "hello"})
        x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
        self.model.fit(x, y)

    def predict(self, idx, structures):
        x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
        return self.model.predict(x)


def test_mofbench(tmp_path_factory):
    """Test the MOFBench class."""
    ds = CoREDataset()
    with pytest.raises(TypeError):
        bench = MOFBench(
            model=MyDummyModel(),
            ds=ds,
            splitter=ClusterStratifiedSplitter(
                CoREDataset(), feature_names=list(ds.available_features)
            ),
            target=["outputs.logKH_CO2"],
        )
    bench = MOFBenchRegression(
        model=MyDummyModel(),
        ds=ds,
        name="my model",
        task="logKH_CO2_id",
        splitter=BaseSplitter(CoREDataset(), sample_frac=0.001),
        target=["outputs.logKH_CO2"],
        debug=True,
        k=2,
    )
    assert isinstance(bench, MOFBench)

    report = bench.bench()
    assert isinstance(report, BenchResult)
    assert len(report.logs) == 2
    assert report.logs[0] == {"info": "hello"}
    assert isinstance(report.json(), str)
    path = os.path.join(tmp_path_factory.mktemp("report"))
    report.save_json(path)
    read_bench = BenchResult.parse_file(
        os.path.join(path, f"{id_for_bench_result(report)}.json"),
    )
    assert isinstance(read_bench, BenchResult)
