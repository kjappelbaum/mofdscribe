# -*- coding: utf-8 -*-
from mofdscribe.utils.raspa.resize_uc import resize_unit_cell


def test_resize_unit_cell(hkust_structure):
    supercell = resize_unit_cell(hkust_structure, 28)
    assert supercell == (2, 2, 2)
