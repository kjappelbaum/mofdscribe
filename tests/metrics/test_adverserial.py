# -*- coding: utf-8 -*-
"""Test the adversial validation"""

import numpy as np

from mofdscribe.datasets import CoREDataset
from mofdscribe.metrics.adverserial import AdverserialValidator
from mofdscribe.splitters import DensitySplitter, HashSplitter


def test_adverserial_validator():
    """Test the adverserial validation"""
    dataset = CoREDataset()
    splitter = HashSplitter(dataset, sample_frac=0.2)

    feature_names = list(dataset.available_features)

    train_idx, test_idx = splitter.train_test_split(frac_train=0.5)

    x_a = dataset._df[feature_names].iloc[train_idx]
    x_b = dataset._df[feature_names].iloc[test_idx]

    adv = AdverserialValidator(x_a, x_b)
    score = adv.score()
    assert len(score) == 5
    assert np.abs(score.mean() - 0.5) < 0.25


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

    feature_names = list(dataset.available_features)

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
