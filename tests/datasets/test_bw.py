# -*- coding: utf-8 -*-
"""Test BW dataset."""
import numpy as np
from pymatgen.core import IStructure

from mofdscribe.datasets.bw_dataset import BWDataset


def test_bw():
    """Ensure we can instantiate the BW dataset and access a few key methods."""
    bw = BWDataset()
    assert isinstance(list(bw.get_structures([1]))[0], IStructure)
    assert isinstance(bw.get_labels([1]), np.ndarray)

    # make sure we can get a subset of the dataset
    bw_subset = bw.get_subset([1, 2, 3, 8])
    assert isinstance(bw_subset, BWDataset)
    assert len(bw_subset._df) == 4
    assert (
        bw._df.iloc[[1, 2, 3, 8]]["info.name"].values == bw_subset._df["info.name"].values
    ).all()
    assert list(bw.get_structures([1]))[0] == list(bw_subset.get_structures([0]))[0]
