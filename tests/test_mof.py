# -*- coding: utf-8 -*-
"""test the MOF base class"""
from pymatgen.analysis.graphs import StructureGraph

from mofdscribe.mof import MOF


def test_mof(hkust_structure):
    mof = MOF(hkust_structure)
    assert isinstance(mof.structure_graph, StructureGraph)
    fragments = mof.fragments
    assert hasattr(fragments, "linkers")
    assert isinstance(mof.decorated_graph_hash, str)
    assert isinstance(mof.decorated_scaffold_hash, str)
    assert isinstance(mof.undecorated_graph_hash, str)
    assert isinstance(mof.undecorated_scaffold_hash, str)


def test_mof_from_file(hkust_path):
    mof = MOF.from_file(hkust_path)
    sg_non_primitive = mof.structure_graph
    assert isinstance(sg_non_primitive, StructureGraph)
    fragments = mof.fragments
    assert hasattr(fragments, "linkers")
    assert isinstance(mof.decorated_graph_hash, str)
    assert isinstance(mof.decorated_scaffold_hash, str)
    assert isinstance(mof.undecorated_graph_hash, str)
    assert isinstance(mof.undecorated_scaffold_hash, str)
