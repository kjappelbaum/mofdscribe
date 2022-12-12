# -*- coding: utf-8 -*-
"""Test thermal stability dataset."""
import numpy as np
import pytest
from pymatgen.core import IStructure

from mofdscribe.datasets.thermal_stability_dataset import ThermalStabilityDataset


@pytest.mark.xdist_group(name="thermal-stability-ds")
def test_thermal_stability():
    dataset = ThermalStabilityDataset()

    assert isinstance(list(dataset.get_structures([1]))[0], IStructure)
    assert isinstance(dataset.get_labels([1]), np.ndarray)
    assert isinstance(dataset.get_years([1]), np.ndarray)
    assert len(dataset.get_years([1])) == 1
    assert len(dataset.get_years([1, 2, 4])) == 3

    # make sure we can get a subset of the dataset
    subset = dataset.get_subset([1, 2, 3, 8])
    assert isinstance(subset, ThermalStabilityDataset)
    assert len(subset._df) == 4
    assert (
        dataset._df.iloc[[1, 2, 3, 8]]["info.basename"].values == subset._df["info.basename"].values
    ).all()
    assert list(dataset.get_structures([1]))[0] == list(subset.get_structures([0]))[0]
