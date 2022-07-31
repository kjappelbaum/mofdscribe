# -*- coding: utf-8 -*-

from pymatgen.core import Element

from mofdscribe.featurizers.chemistry._fragment import (
    get_bb_indices,
    get_bbs_from_indices,
    get_floating_indices,
    get_node_atoms,
)
from mofdscribe.featurizers.utils.structure_graph import get_structure_graph


def test_get_floating_indices(hkust_structure, irmof_structure, abacuf_structure):
    for structure in [hkust_structure, irmof_structure, abacuf_structure]:
        sg = get_structure_graph(structure, "vesta")

        assert len(get_floating_indices(sg)) == 0


def test_get_node_atoms(hkust_structure):
    sg = get_structure_graph(hkust_structure, "vesta")
    node_atoms = get_node_atoms(sg)
    assert len(node_atoms) < len(sg)
    assert isinstance(node_atoms, set)
    for site in node_atoms:
        assert str(hkust_structure[site].specie) == "Cu"
    assert len(node_atoms) == dict(hkust_structure.composition)[Element("Cu")]


def test_get_bbs_from_indices(hkust_structure):
    sg = get_structure_graph(hkust_structure, "vesta")
    node_atoms = get_node_atoms(sg)
    all_atoms = set(list(range(len(sg))))
    linker_atoms = all_atoms - node_atoms
    assert linker_atoms & node_atoms == set()
    (
        linker_mol,
        linker_subgraph,
        linker_idx,
        linker_center,
        linker_coordinate,
    ) = get_bbs_from_indices(sg, linker_atoms)
    flat_linker_idx = sum(linker_idx, [])
    assert set(flat_linker_idx) & set(node_atoms) == set()
    assert len(sg) == len(node_atoms) + len(set(flat_linker_idx))
    for idx in flat_linker_idx:
        assert str(hkust_structure[idx].specie) in ("H", "C", "O", "N")


def test_get_bb_indices(hkust_structure):
    sg = get_structure_graph(hkust_structure, "vesta")
    bb_indices = get_bb_indices(sg)
    assert len(bb_indices["nodes"]) < len(sg)
    assert len(bb_indices["nodes"]) > 0
    assert (
        len(set(sum(bb_indices["nodes"], []))) == dict(hkust_structure.composition)[Element("Cu")]
    )

    for site in sum(bb_indices["linker_all"], []):
        assert str(hkust_structure[site].specie) in ("H", "C", "O", "N")

    assert (
        len(set(sum(bb_indices["linker_all"], [])))
        == len(sg) - dict(hkust_structure.composition)[Element("Cu")]
    )

    assert (
        len(sg) - len(set(sum(bb_indices["nodes"], []))) - len(sum(bb_indices["linker_all"], []))
        == 0
    )

    assert len(sum(bb_indices["linker_functional"], [])) == 0
    assert len(sum(bb_indices["linker_all"], [])) == len(
        sum(bb_indices["linker_scaffold"], [])
    ) + len(sum(bb_indices["linker_connecting"], []))
