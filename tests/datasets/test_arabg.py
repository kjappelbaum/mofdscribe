# -*- coding: utf-8 -*-
"""Test ARABG dataset."""
import numpy as np
from pymatgen.core import IStructure

from mofdscribe.datasets.arabg_dataset import ARABGDataset


def test_arabg():
    """Ensure we can instantiate the ARABG dataset and access a few key methods."""
    arabg = ARABGDataset()
    assert isinstance(list(arabg.get_structures([1]))[0], IStructure)
    assert isinstance(arabg.get_labels([1]), np.ndarray)

    # make sure we can get a subset of the dataset
    arabg_subset = arabg.get_subset([1, 2, 3, 8])
    assert isinstance(arabg_subset, ARABGDataset)
    assert len(arabg_subset._df) == 4
    assert (
        arabg._df.iloc[[1, 2, 3, 8]]["info.name"].values == arabg_subset._df["info.name"].values
    ).all()
    assert list(arabg.get_structures([1]))[0] == list(arabg_subset.get_structures([0]))[0]
