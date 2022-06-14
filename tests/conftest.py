# -*- coding: utf-8 -*-
import os

import pytest
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.structure_graph import get_structure_graph

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def hkust_structure():
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))


@pytest.fixture
def irmof_structure():
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "IRMOF-1.cif"))


@pytest.fixture
def cof_structure():
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "20450N2_ddec.cif"))


@pytest.fixture
def hkust_structure_graph():
    return get_structure_graph(
        IStructure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    )
