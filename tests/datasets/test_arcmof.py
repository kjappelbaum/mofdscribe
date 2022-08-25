# -*- coding: utf-8 -*-
"""Test ARCMOF dataset."""
import numpy as np
from pymatgen.core import IStructure

from mofdscribe.datasets.arcmof_dataset import ARCMOFDataset


def test_arcmof():
    """Ensure we can instantiate the ARCMOF dataset and access a few key methods."""
    arc = ARCMOFDataset()
    assert isinstance(list(arc.get_structures([1]))[0], IStructure)
    assert isinstance(arc.get_labels([1]), np.ndarray)

    # make sure we can get a subset of the dataset
    arc_subset = arc.get_subset([1, 2, 3, 8])
    assert isinstance(arc_subset, ARCMOFDataset)
    assert len(arc_subset._df) == 4
    assert (
        arc._df.iloc[[1, 2, 3, 8]]["info.name"].values == arc_subset._df["info.name"].values
    ).all()
    assert list(arc.get_structures([1]))[0] == list(arc_subset.get_structures([0]))[0]
