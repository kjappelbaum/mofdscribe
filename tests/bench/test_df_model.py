# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestRegressor

from mofdscribe.bench.df_model import DFModel
from mofdscribe.bench.logkHCO2 import _FEATURES, LogkHCO2ExtrapolationBench
from mofdscribe.bench.mofbench import BenchResult
from mofdscribe.datasets import CoREDataset


def test_df_model():
    X = CoREDataset()._df[_FEATURES]  # noqa: N806
    model = RandomForestRegressor(n_estimators=100)
    df_model = DFModel(model, X)
    bench = LogkHCO2ExtrapolationBench(df_model, name="bla")
    res = bench.bench()
    assert isinstance(res, BenchResult)
