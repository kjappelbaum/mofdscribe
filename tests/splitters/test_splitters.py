# -*- coding: utf-8 -*-
from mofdscribe.datasets.core_dataset import CoREDataset
from mofdscribe.splitters.splitters import (
    DensitySpliiter,
    HashSplitter,
    TimeSplitter,
    RandomSplitter,
    FingerprintSplitter,
)


def test_hash_splitter():
    ds = CoREDataset()
    hs = HashSplitter(hash_type="undecorated_scaffold_hash")

    for train, test in hs.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)

    splits = hs.train_valid_test_split(ds, frac_train=0.5, frac_valid=0.3, shuffle=True)
    assert len(splits) == 3
    assert len(splits[0]) > len(splits[1]) > len(splits[2])


def test_time_splitter():
    ds = CoREDataset()
    ts = TimeSplitter()

    for train, test in ts.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)

    splits = ts.train_valid_test_split(ds, frac_train=0.5, frac_valid=0.3, shuffle=True)
    assert len(splits) == 3
    assert len(splits[0]) > len(splits[1]) > len(splits[2])


def test_density_splitter():
    ds = CoREDataset()

    dens_splitter = DensitySpliiter()
    for train, test in dens_splitter.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)

    splits = dens_splitter.train_valid_test_split(ds, frac_train=0.5, frac_valid=0.3, shuffle=True)
    assert len(splits) == 3
    assert len(splits[0]) > len(splits[1]) > len(splits[2])


def test_random_splitter():
    ds = CoREDataset()

    rs = RandomSplitter()
    for train, test in rs.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)

    splits = rs.train_valid_test_split(ds, frac_train=0.5, frac_valid=0.3, shuffle=True)
    assert len(splits) == 3
    assert len(splits[0]) > len(splits[1]) > len(splits[2])


def test_fingerprint_splitter():
    ds = CoREDataset()

    fps = FingerprintSplitter(
        feature_names=[
            "total_POV_gravimetric",
            "mc_CRY-chi-0-all",
            "mc_CRY-chi-1-all",
            "mc_CRY-chi-2-all",
            "mc_CRY-chi-3-all",
            "mc_CRY-Z-0-all",
            "mc_CRY-Z-1-all",
            "mc_CRY-Z-2-all",
            "mc_CRY-Z-3-all",
            "mc_CRY-I-0-all",
            "mc_CRY-I-1-all",
            "mc_CRY-I-2-all",
            "mc_CRY-I-3-all",
            "mc_CRY-T-0-all",
            "mc_CRY-T-1-all",
            "mc_CRY-T-2-all",
            "mc_CRY-T-3-all",
            "mc_CRY-S-0-all",
            "mc_CRY-S-1-all",
            "mc_CRY-S-2-all",
            "mc_CRY-S-3-all",
            "D_mc_CRY-chi-0-all",
            "D_mc_CRY-chi-1-all",
            "D_mc_CRY-chi-2-all",
            "D_mc_CRY-chi-3-all",
            "D_mc_CRY-Z-0-all",
            "D_mc_CRY-Z-1-all",
            "D_mc_CRY-Z-2-all",
            "D_mc_CRY-Z-3-all",
            "D_mc_CRY-I-0-all",
            "D_mc_CRY-I-1-all",
            "D_mc_CRY-I-2-all",
            "D_mc_CRY-I-3-all",
            "D_mc_CRY-T-0-all",
            "D_mc_CRY-T-1-all",
            "D_mc_CRY-T-2-all",
            "D_mc_CRY-T-3-all",
            "D_mc_CRY-S-0-all",
            "D_mc_CRY-S-1-all",
            "D_mc_CRY-S-2-all",
            "D_mc_CRY-S-3-all",
            "sum-mc_CRY-chi-0-all",
            "sum-mc_CRY-chi-1-all",
            "sum-mc_CRY-chi-2-all",
            "sum-mc_CRY-chi-3-all",
            "sum-mc_CRY-Z-0-all",
            "sum-mc_CRY-Z-1-all",
            "sum-mc_CRY-Z-2-all",
            "sum-mc_CRY-Z-3-all",
            "sum-mc_CRY-I-0-all",
            "sum-mc_CRY-I-1-all",
            "sum-mc_CRY-I-2-all",
            "sum-mc_CRY-I-3-all",
            "sum-mc_CRY-T-0-all",
            "sum-mc_CRY-T-1-all",
            "sum-mc_CRY-T-2-all",
            "sum-mc_CRY-T-3-all",
            "sum-mc_CRY-S-0-all",
            "sum-mc_CRY-S-1-all",
            "sum-mc_CRY-S-2-all",
            "sum-mc_CRY-S-3-all",
            "sum-D_mc_CRY-chi-0-all",
            "sum-D_mc_CRY-chi-1-all",
            "sum-D_mc_CRY-chi-2-all",
            "sum-D_mc_CRY-chi-3-all",
            "sum-D_mc_CRY-Z-0-all",
            "sum-D_mc_CRY-Z-1-all",
            "sum-D_mc_CRY-Z-2-all",
            "sum-D_mc_CRY-Z-3-all",
            "sum-D_mc_CRY-I-0-all",
            "sum-D_mc_CRY-I-1-all",
            "sum-D_mc_CRY-I-2-all",
            "sum-D_mc_CRY-I-3-all",
            "sum-D_mc_CRY-T-0-all",
            "sum-D_mc_CRY-T-1-all",
            "sum-D_mc_CRY-T-2-all",
            "sum-D_mc_CRY-T-3-all",
            "sum-D_mc_CRY-S-0-all",
            "sum-D_mc_CRY-S-1-all",
            "sum-D_mc_CRY-S-2-all",
            "sum-D_mc_CRY-S-3-all",
            "D_lc-chi-0-all",
            "D_lc-chi-1-all",
            "D_lc-chi-2-all",
            "D_lc-chi-3-all",
            "D_lc-Z-0-all",
            "D_lc-Z-1-all",
            "D_lc-Z-2-all",
            "D_lc-Z-3-all",
            "D_lc-I-0-all",
            "D_lc-I-1-all",
            "D_lc-I-2-all",
            "D_lc-I-3-all",
            "D_lc-T-0-all",
            "D_lc-T-1-all",
            "D_lc-T-2-all",
            "D_lc-T-3-all",
            "D_lc-S-0-all",
            "D_lc-S-1-all",
            "D_lc-S-2-all",
            "D_lc-S-3-all",
            "D_lc-alpha-0-all",
            "D_lc-alpha-1-all",
            "D_lc-alpha-2-all",
            "D_lc-alpha-3-all",
            "D_func-chi-0-all",
            "D_func-chi-1-all",
            "D_func-chi-2-all",
            "D_func-chi-3-all",
            "D_func-Z-0-all",
            "D_func-Z-1-all",
            "D_func-Z-2-all",
            "D_func-Z-3-all",
            "D_func-I-0-all",
            "D_func-I-1-all",
            "D_func-I-2-all",
            "D_func-I-3-all",
            "D_func-T-0-all",
            "D_func-T-1-all",
            "D_func-T-2-all",
            "D_func-T-3-all",
            "D_func-S-0-all",
            "D_func-S-1-all",
            "D_func-S-2-all",
            "D_func-S-3-all",
            "D_func-alpha-0-all",
            "D_func-alpha-1-all",
            "D_func-alpha-2-all",
            "D_func-alpha-3-all",
            "sum-D_lc-chi-0-all",
            "sum-D_lc-chi-1-all",
            "sum-D_lc-chi-2-all",
            "sum-D_lc-chi-3-all",
            "sum-D_lc-Z-0-all",
            "sum-D_lc-Z-1-all",
            "sum-D_lc-Z-2-all",
            "sum-D_lc-Z-3-all",
            "sum-D_lc-I-0-all",
            "sum-D_lc-I-1-all",
            "sum-D_lc-I-2-all",
            "sum-D_lc-I-3-all",
            "sum-D_lc-T-0-all",
            "sum-D_lc-T-1-all",
            "sum-D_lc-T-2-all",
            "sum-D_lc-T-3-all",
            "sum-D_lc-S-0-all",
            "sum-D_lc-S-1-all",
            "sum-D_lc-S-2-all",
            "sum-D_lc-S-3-all",
            "sum-D_lc-alpha-0-all",
            "sum-D_lc-alpha-1-all",
            "sum-D_lc-alpha-2-all",
            "sum-D_lc-alpha-3-all",
            "sum-D_func-chi-0-all",
            "sum-D_func-chi-1-all",
            "sum-D_func-chi-2-all",
            "sum-D_func-chi-3-all",
            "sum-D_func-Z-0-all",
            "sum-D_func-Z-1-all",
            "sum-D_func-Z-2-all",
            "sum-D_func-Z-3-all",
            "sum-D_func-I-0-all",
            "sum-D_func-I-1-all",
            "sum-D_func-I-2-all",
            "sum-D_func-I-3-all",
            "sum-D_func-T-0-all",
            "sum-D_func-T-1-all",
            "sum-D_func-T-2-all",
            "sum-D_func-T-3-all",
            "sum-D_func-S-0-all",
            "sum-D_func-S-1-all",
            "sum-D_func-S-2-all",
            "sum-D_func-S-3-all",
            "sum-D_func-alpha-0-all",
            "sum-D_func-alpha-1-all",
            "sum-D_func-alpha-2-all",
            "sum-D_func-alpha-3-all",
        ]
    )
    for train, test in fps.k_fold(ds, k=5, shuffle=True):

        assert set(train) & set(test) == set()
        assert len(set(list(train) + list(test))) == len(ds)

    splits = fps.train_valid_test_split(ds, frac_train=0.5, frac_valid=0.3, shuffle=True)
    assert len(splits) == 3
    assert len(splits[0]) > len(splits[1]) > len(splits[2])
