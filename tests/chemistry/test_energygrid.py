# -*- coding: utf-8 -*-
import os

from mofdscribe.chemistry.energygrid import EnergyGridHistogram, read_ascii_grid

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_ascii_grid():
    file_name = os.path.join(THIS_DIR, "..", "test_files", "asci_grid_C_co2.grid")
    result = read_ascii_grid(file_name)
    assert len(result) == 22185
    assert result["energy"].dtype == float


def test_energygrid(hkust_structure):
    eg = EnergyGridHistogram()
    feats = eg.featurize(hkust_structure)

    assert len(feats) == 40
    assert len(eg.feature_labels()) == 40
