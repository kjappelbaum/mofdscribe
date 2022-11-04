# -*- coding: utf-8 -*-
"""Test fixtures"""
import json
import os
from glob import glob

import pytest
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import IStructure, Molecule, Structure
from structuregraph_helpers.create import get_structure_graph

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def hkust_structure():
    """Return a pymatgen Structure for HKUST"""
    return IStructure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))


@pytest.fixture(scope="session")
def hkust_la_structure():
    """Return a pymatgen Structure for HKUST"""
    return IStructure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1-La.cif"))


@pytest.fixture(scope="session")
def abacuf_structure():
    """Return a pymatgen Structure for ABACUF"""
    return IStructure.from_file(os.path.join(THIS_DIR, "test_files", "ABACUF.cif"))


@pytest.fixture(scope="session")
def irmof_structure():
    """Return a pymatgen Structure for IRMOF"""
    return IStructure.from_file(os.path.join(THIS_DIR, "test_files", "IRMOF-1.cif"))


@pytest.fixture(scope="session")
def cof_structure():
    """Return a pymatgen Structure for a COF"""
    return IStructure.from_file(os.path.join(THIS_DIR, "test_files", "20450N2_ddec.cif"))


@pytest.fixture(scope="session")
def hkust_structure_graph():
    """Return a pymatgen StructureGraph for HKUST"""
    return get_structure_graph(
        IStructure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    )


@pytest.fixture(scope="session")
def molecule_graph():
    """Return a pymatgen MoleculeGraph for a HKUST node"""
    with open(os.path.join(THIS_DIR, "test_files", "test_molecule_graph.json"), "r") as handle:
        mol_graph = MoleculeGraph.from_dict(json.loads(handle.read()))
    return mol_graph


@pytest.fixture(scope="session")
def molecule():
    """Return a pymatgen Molecule for a HKUST node"""
    with open(os.path.join(THIS_DIR, "test_files", "test_molecule.json"), "r") as handle:
        mol = Molecule.from_dict(json.loads(handle.read()))
    return mol


@pytest.fixture(scope="session")
def linker_molecule():
    """Return a pymatgen Molecule for BTC linker"""
    with open(os.path.join(THIS_DIR, "test_files", "linker_molecule.json"), "r") as handle:
        mol = Molecule.from_dict(json.loads(handle.read()))
    return mol


@pytest.fixture(scope="session")
def triangle_structure():
    """Return a pymatgen Structure for the `connecting sites` of a BTC linker"""
    with open(os.path.join(THIS_DIR, "test_files", "triangle_structure.json"), "r") as handle:
        s = IStructure.from_dict(json.loads(handle.read()))
    return s


@pytest.fixture(scope="session")
def floating_structure():
    """Return IRMOF with floating mol"""
    return IStructure.from_file(os.path.join(THIS_DIR, "test_files", "floating_check.cif"))


@pytest.fixture(scope="session")
def hkust_linker_structure():
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "BTC.cif"))


@pytest.fixture(scope="session")
def dataset_files():
    structures = glob(os.path.join(THIS_DIR, "test_files", "test_dataset", "structures", "*.cif"))
    dataset = glob(os.path.join(THIS_DIR, "test_files", "test_dataset", "data.json"))
    return structures, dataset


@pytest.fixture(scope="session")
def dataset_folder():
    return os.path.join(THIS_DIR, "test_files", "test_dataset")
