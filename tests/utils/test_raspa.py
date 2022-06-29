# -*- coding: utf-8 -*-
"""Test the helpers for running simulations with RASPA"""
from mofdscribe.featurizers.utils.raspa.resize_uc import resize_unit_cell


def test_resize_unit_cell(hkust_structure):
    """Test the resize_unit_cell function"""
    supercell = resize_unit_cell(hkust_structure, 28)
    assert supercell == (2, 2, 2)
