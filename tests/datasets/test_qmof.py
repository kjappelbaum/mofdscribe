# -*- coding: utf-8 -*-
"""Test QMOF dataset."""
import numpy as np
from pymatgen.core import IStructure

from mofdscribe.datasets.qmof_dataset import QMOFDataset


def test_core():
    """Ensure we can instantiate the QMOF dataset and access a few key methods."""
    qmof = QMOFDataset()
    assert isinstance(list(qmof.get_structures([1]))[0], IStructure)
    assert isinstance(qmof.get_labels([1]), np.ndarray)
    assert isinstance(qmof.get_years([1]), np.ndarray)
    assert len(qmof.get_years([1])) == 1
    assert len(qmof.get_years([1, 2, 4])) == 3
    assert isinstance(qmof.get_labels([1], ['outputs.pbe.bandgap']), np.ndarray)
