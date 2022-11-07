# -*- coding: utf-8 -*-
"""Test CoRE dataset."""
import numpy as np
import pytest
from pymatgen.core import IStructure

from mofdscribe.datasets.core_dataset import CoREDataset


@pytest.mark.xdist_group(name="core-ds")
def test_core():
    """Ensure we can instantiate the CoRE dataset as access a few key methods."""
    core = CoREDataset()
    assert isinstance(list(core.get_structures([1]))[0], IStructure)
    assert isinstance(core.get_labels([1]), np.ndarray)
    assert isinstance(core.get_years([1]), np.ndarray)
    assert len(core.get_years([1])) == 1
    assert len(core.get_years([1, 2, 4])) == 3
    assert isinstance(core.get_labels([1], ["outputs.pure_methane_kH"]), np.ndarray)

    # make sure we can get a subset of the dataset
    core_subset = core.get_subset([1, 2, 3, 8])
    assert isinstance(core_subset, CoREDataset)
    assert len(core_subset._df) == 4
    assert (
        core._df.iloc[[1, 2, 3, 8]]["info.basename"].values
        == core_subset._df["info.basename"].values
    ).all()
    assert list(core.get_structures([1]))[0] == list(core_subset.get_structures([0]))[0]
