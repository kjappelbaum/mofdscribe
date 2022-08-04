# -*- coding: utf-8 -*-
"""Test the EnergyGridHistogram featurizer."""
import os

from mofdscribe.featurizers.chemistry.energygrid import EnergyGridHistogram, read_ascii_grid

from ..helpers import is_jsonable

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_ascii_grid():
    """Ensure that we can parse an ASCII grid file."""
    file_name = os.path.join(THIS_DIR, "..", "..", "test_files", "asci_grid_C_co2.grid")
    result = read_ascii_grid(file_name)
    assert len(result) == 22185
    assert result["energy"].dtype == float


def test_energygrid(hkust_structure):
    """Make sure that the featurization works for typical MOFs and the number of features is as expected."""
    eg = EnergyGridHistogram()
    feats = eg.featurize(hkust_structure)

    assert len(feats) == 40
    assert len(eg.feature_labels()) == 40
    assert is_jsonable(dict(zip(eg.feature_labels(), feats)))
    assert feats.ndim == 1
