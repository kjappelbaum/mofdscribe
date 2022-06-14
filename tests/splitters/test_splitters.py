# -*- coding: utf-8 -*-
from mofdscribe.datasets.core_dataset import CoREDataset
from mofdscribe.splitters.splitters import DensitySpliiter, HashSplitter, TimeSplitter


def test_hash_splitter():
    ds = CoREDataset()
    hs = HashSplitter(hash_type="undecorated_scaffold_hash")

    for train, test in hs.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)


def test_time_splitter():
    ds = CoREDataset()
    ts = TimeSplitter()

    for train, test in ts.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)


def test_density_splitter():
    ds = CoREDataset()

    dens_splitter = DensitySpliiter()
    for train, test in dens_splitter.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)
