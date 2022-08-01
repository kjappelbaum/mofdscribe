# -*- coding: utf-8 -*-
"""Test the adversial validation"""
from tkinter import Spinbox

import numpy as np

from mofdscribe.datasets import CoREDataset
from mofdscribe.metrics.adverserial import AdverserialValidator
from mofdscribe.splitters import DensitySplitter, HashSplitter


def test_adverserial_validator():
    """Test the adverserial validation"""
    dataset = CoREDataset()
    splitter = HashSplitter(dataset, sample_frac=0.2)

    feature_names = [
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

    train_idx, test_idx = splitter.train_test_split(frac_train=0.5)

    x_a = dataset._df[feature_names].iloc[train_idx]
    x_b = dataset._df[feature_names].iloc[test_idx]

    adv = AdverserialValidator(x_a, x_b)
    score = adv.score()
    assert len(score) == 5
    assert np.abs(score.mean() - 0.5) < 0.1


def test_adverserial_validator_with_different_dist():
    """We use the DensitySplitter to create different distributions.

    Make sure that we detect this and highlight reasonable features.
    """
    dataset = CoREDataset()
    splitter = DensitySplitter(dataset, sample_frac=0.5, density_q=[0, 0.5, 1])

    densities = dataset.get_densities(range(len(dataset)))
    groups = splitter._get_groups()

    assert (
        np.abs(densities[groups == 0].mean() - densities[groups == 1].mean())
        > densities[groups == 0].std()
    )

    feature_names = [
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
    ]

    train_idx, test_idx = splitter.train_test_split(frac_train=0.5)

    x_a = dataset._df[feature_names].iloc[train_idx]
    x_b = dataset._df[feature_names].iloc[test_idx]

    assert (
        np.abs(densities[train_idx].mean() - densities[test_idx].mean())
        > densities[train_idx].std()
    )
    adv = AdverserialValidator(x_a, x_b)
    score = adv.score()
    assert len(score) == 5
    assert np.abs(score.mean() - 0.5) > 0.3

    assert np.argmax(adv.get_feature_importance()) == 0
