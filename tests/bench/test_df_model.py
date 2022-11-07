# -*- coding: utf-8 -*-
import pytest
from sklearn.ensemble import RandomForestRegressor

from mofdscribe.bench.df_model import DFModel
from mofdscribe.bench.logkHCO2 import LogkHCO2OODBench
from mofdscribe.bench.mofbench import BenchResult
from mofdscribe.datasets import CoREDataset


@pytest.mark.xdist_group(name="core-ds")
def test_df_model():
    ds = CoREDataset()
    model = RandomForestRegressor(n_estimators=100)
    df_model = DFModel(model, features=list(ds.available_features))
    bench = LogkHCO2OODBench(df_model, name="bla", debug=True, patch_in_ds=True)
    res = bench.bench()
    assert isinstance(res, BenchResult)
