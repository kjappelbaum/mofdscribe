# -*- coding: utf-8 -*-
"""Test fixtures"""
import json
import os

import pytest
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import IStructure, Molecule, Structure

from mofdscribe.utils.structure_graph import get_structure_graph

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def hkust_structure():
    """Return a pymatgen Structure for HKUST"""
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))


@pytest.fixture
def irmof_structure():
    """Return a pymatgen Structure for IRMOF"""
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "IRMOF-1.cif"))


@pytest.fixture
def cof_structure():
    """Return a pymatgen Structure for a COF"""
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "20450N2_ddec.cif"))


@pytest.fixture
def hkust_structure_graph():
    """Return a pymatgen StructureGraph for HKUST"""
    return get_structure_graph(
        IStructure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    )


@pytest.fixture
def molecule_graph():
    """Return a pymatgen MoleculeGraph for a HKUST node"""
    with open(os.path.join(THIS_DIR, "test_files", "test_molecule_graph.json"), "r") as handle:
        mol_graph = MoleculeGraph.from_dict(json.loads(handle.read()))
    return mol_graph


@pytest.fixture
def molecule():
    """Return a pymatgen Molecule for a HKUST node"""
    with open(os.path.join(THIS_DIR, "test_files", "test_molecule.json"), "r") as handle:
        mol = Molecule.from_dict(json.loads(handle.read()))
    return mol
