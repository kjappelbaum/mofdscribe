# -*- coding: utf-8 -*-
"""Test the splitter helping functions."""

from random import shuffle

import numpy as np
import pytest

from mofdscribe.splitters.utils import (
    check_fraction,
    downsample_splits,
    get_train_valid_test_sizes,
    grouped_stratified_train_test_partition,
    grouped_train_valid_test_partition,
    kennard_stone_sampling,
    stratified_train_test_partition,
)


def test_kennard_stone_sampling():
    """Ensure we get the order we would expect."""
    X = np.array([[1, 2, 3], [4, 5, 6], [8, 8, 9]])  # noqa: N806

    indices = kennard_stone_sampling(X)
    assert indices == [2, 0, 1]

    # Make sure also the other options do not complain
    indices = kennard_stone_sampling(X, centrality_measure="median")
    assert indices == [2, 0, 1]

    indices = kennard_stone_sampling(X, centrality_measure="random")
    assert len(indices) == 3  # we cannot guarantee the order of the indices


@pytest.mark.parametrize("number_of_groups", [30, 50, 80, 500, 8_000])
@pytest.mark.parametrize("shuffle", [True])
def test_grouped_stratified_train_test_partition(number_of_groups, shuffle):
    # perhaps use hypothesis for fuzzing the data
    datasize = 10_000

    y = np.random.normal(0, 1, size=datasize)
    groups = np.random.choice(np.arange(number_of_groups), size=datasize)

    train_indices, valid_indices, test_indices = grouped_stratified_train_test_partition(
        y, groups, train_size=0.5, valid_size=0.25, test_size=0.25, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) + len(valid_indices) == datasize

    # make sure we have no index overlap
    assert len(np.intersect1d(train_indices, valid_indices)) == 0
    assert len(np.intersect1d(train_indices, test_indices)) == 0

    # make sure we have no group overlap
    test_groups = groups[test_indices]
    train_groups = groups[train_indices]
    valid_groups = groups[valid_indices]
    assert set(test_groups).intersection(set(train_groups)) == set()
    assert set(test_groups).intersection(set(valid_groups)) == set()
    assert set(train_groups).intersection(set(valid_groups)) == set()

    # since we stratify we should have comparable medians
    train_mean = np.median(y[train_indices])
    valid_mean = np.median(y[valid_indices])
    test_mean = np.median(y[test_indices])
    assert (
        pytest.approx(train_mean, abs=0.1)
        == pytest.approx(valid_mean, abs=0.1)
        == pytest.approx(test_mean, abs=0.1)
    )

    # now, no validation set
    train_indices, _, test_indices = grouped_stratified_train_test_partition(
        y, groups, train_size=0.8, valid_size=0.0, test_size=0.2, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) == datasize

    test_groups = groups[test_indices]
    train_groups = groups[train_indices]

    assert set(test_groups).intersection(set(train_groups)) == set()


def test_stratified_train_test_partition():
    # perhaps use hypothesis for fuzzing the data
    datasize = 10_000

    y = np.random.normal(0, 1, size=datasize)

    train_indices, valid_indices, test_indices = stratified_train_test_partition(
        np.arange(datasize), y, train_size=0.5, valid_size=0.25, test_size=0.25, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) + len(valid_indices) == datasize

    # make sure we have no index overlap
    assert len(np.intersect1d(train_indices, valid_indices)) == 0
    assert len(np.intersect1d(train_indices, test_indices)) == 0

    # since we stratify we should have comparable medians
    train_mean = np.median(y[train_indices])
    valid_mean = np.median(y[valid_indices])
    test_mean = np.median(y[test_indices])
    assert (
        pytest.approx(train_mean, abs=0.1)
        == pytest.approx(valid_mean, abs=0.1)
        == pytest.approx(test_mean, abs=0.1)
    )

    # now, no validation set
    train_indices, _, test_indices = stratified_train_test_partition(
        np.arange(datasize), y, train_size=0.8, valid_size=0.0, test_size=0.2, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) == datasize

    assert len(np.intersect1d(train_indices, test_indices)) == 0


@pytest.mark.parametrize("number_of_groups", [30, 50, 80, 500, 8_000])
@pytest.mark.parametrize("shuffle", [True, False])
def test_grouped_train_valid_test_partition(number_of_groups, shuffle):
    # we might also want to use grouping without stratification to test
    # extrapolation
    datasize = 10_000
    groups = np.random.choice(np.arange(number_of_groups), size=datasize)

    train_indices, valid_indices, test_indices = grouped_train_valid_test_partition(
        groups, train_size=0.5, valid_size=0.25, test_size=0.25, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) + len(valid_indices) == datasize
    test_groups = groups[test_indices]
    train_groups = groups[train_indices]
    valid_groups = groups[valid_indices]
    assert set(test_groups).intersection(set(train_groups)) == set()
    assert set(test_groups).intersection(set(valid_groups)) == set()
    assert set(train_groups).intersection(set(valid_groups)) == set()

    train_indices, valid_indices, test_indices = grouped_train_valid_test_partition(
        groups, train_size=0.5, valid_size=0, test_size=0.5, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) == datasize
    test_groups = groups[test_indices]
    train_groups = groups[train_indices]

    assert set(test_groups).intersection(set(train_groups)) == set()


@pytest.mark.parametrize("shuffle", [True])
def test_grouped_train_valid_test_partition_string_groups(shuffle):
    datasize = 10_000
    groups = np.random.choice(["a", "b", "c", "d", "e", "f"], size=datasize)
    train_indices, valid_indices, test_indices = grouped_train_valid_test_partition(
        groups, train_size=0.5, valid_size=0.25, test_size=0.25, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) + len(valid_indices) == datasize
    test_groups = groups[test_indices]
    train_groups = groups[train_indices]
    valid_groups = groups[valid_indices]
    assert set(test_groups).intersection(set(train_groups)) == set()
    assert set(test_groups).intersection(set(valid_groups)) == set()
    assert set(train_groups).intersection(set(valid_groups)) == set()

    train_indices, valid_indices, test_indices = grouped_train_valid_test_partition(
        groups, train_size=0.5, valid_size=0, test_size=0.5, shuffle=shuffle
    )

    assert len(train_indices) + len(test_indices) == datasize
    test_groups = groups[test_indices]
    train_groups = groups[train_indices]

    assert set(test_groups).intersection(set(train_groups)) == set()


def test_get_train_valid_test_sizes():
    train_size, valid_size, test_size = get_train_valid_test_sizes(100, 0.5, 0.25, 0.25)
    assert train_size + valid_size + test_size == 100
    assert train_size + valid_size == 75


def test_check_fraction():
    with pytest.raises(ValueError):
        check_fraction(-1, 0, 0)

    with pytest.raises(ValueError):
        check_fraction(1.1, 1.0, 0)

    with pytest.raises(ValueError):
        check_fraction(0.9, 0.2, 0.1)

    with pytest.raises(ValueError):
        check_fraction(0.1, 0.2, 0.4)

    _ = check_fraction(0.8, 0.1, 0.1)


def test_downsample_splits():
    res = downsample_splits([np.arange(100), np.arange(100)], sample_frac=0.8)

    assert len(res[0]) == 80
    assert len(res[1]) == 80
