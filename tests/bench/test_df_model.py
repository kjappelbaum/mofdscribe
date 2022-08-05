# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestRegressor

from mofdscribe.bench.df_model import DFModel
from mofdscribe.bench.logkHCO2 import LogkHCO2OODBench
from mofdscribe.bench.mofbench import BenchResult
from mofdscribe.datasets import CoREDataset


def test_df_model():
    ds = CoREDataset()
    X = ds._df[list(ds.available_features)].fillna(0)  # noqa: N806
    model = RandomForestRegressor(n_estimators=100)
    df_model = DFModel(model, X)
    bench = LogkHCO2OODBench(df_model, name="bla", debug=True)
    res = bench.bench()
    assert isinstance(res, BenchResult)
