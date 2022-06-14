# -*- coding: utf-8 -*-
import numpy as np
from pymatgen.core import IStructure

from mofdscribe.datasets.core_dataset import CoREDataset


def test_core():
    core = CoREDataset()
    assert isinstance(list(core.get_structures([1]))[0], IStructure)
    assert isinstance(core.get_labels([1]), np.ndarray)
    assert isinstance(core.get_years([1]), np.ndarray)
    assert len(core.get_years([1])) == 1
    assert len(core.get_years([1, 2, 4])) == 3
    assert isinstance(core.get_labels([1], ["pure_methane_kH"]), np.ndarray)
