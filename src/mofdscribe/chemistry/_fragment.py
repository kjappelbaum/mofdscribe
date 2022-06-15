# -*- coding: utf-8 -*-
"""Naive fragmentation implementation following Moosavi et al.
which does not require multiple loops over the graph.
For alternative (slower) implementation see MOFfragmentor"""


from copy import deepcopy
from typing import Dict, List, Set

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure

from mofdscribe.utils.structure_graph import get_connected_site_indices, get_subgraphs_as_molecules
from mofdscribe.utils.substructures import _not_relevant_structure_indices, get_metal_indices


def get_node_atoms(structure_graph: StructureGraph) -> Set[int]:
    """Get the indices of the atoms that are assigned as node by identifying the
    metals and their connected atoms."""
    metal_indices = get_metal_indices(structure_graph.structure)
    metal_names = [str(structure_graph.structure[i].specie.symbol) for i in metal_indices]

    node_set = set()
    node_set.update(metal_indices)

    tmp_node_set = set()
    for metal_index in metal_indices:
        tmp_node_set.add(metal_index)
        bonded_to_metal = get_connected_site_indices(structure_graph, metal_index)
        tmp_node_set.update(bonded_to_metal)

    # add atoms that are only connected to metal or hydrogen to the node list + hydrogen atoms connected to them
    for node_atom_index in tmp_node_set:
        all_bonded_atoms = get_connected_site_indices(structure_graph, node_atom_index)
        only_bonded_metal_hydrogen = True
        for index in all_bonded_atoms:
            if not ((str(structure_graph.structure[index].specie) == "H") or index in metal_names):
                only_bonded_metal_hydrogen = False
        if only_bonded_metal_hydrogen:
            node_set.add(node_atom_index)

    final_node_atom_set = deepcopy(node_set)
    for atom_index in node_set:
        for index in get_connected_site_indices(structure_graph, atom_index):
            if str(structure_graph.structure[index].specie) == "H":
                final_node_atom_set.update(index)

    return final_node_atom_set


def get_floating_indices(structure_graph: StructureGraph) -> Set[int]:
    """Get the indices of floating (solvent) molecules in the structure."""
    _, _, idx, _, _ = get_subgraphs_as_molecules(structure_graph)
    return set(idx)


def get_bbs_from_indices(structure_graph: StructureGraph, indices: Set[int]):
    graph_ = structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(graph_.structure, indices)
    graph_.remove_nodes(to_delete)
    mol, return_subgraphs, idx, centers, coordinates = get_subgraphs_as_molecules(
        graph_, return_unique=False
    )
    return mol, return_subgraphs, idx, centers, coordinates


def get_bb_indices(structure_graph: StructureGraph) -> Dict[str, List[List[int]]]:
    node_atoms = get_node_atoms(structure_graph)
    floating_indices = get_floating_indices(structure_graph)

    all_atoms = set(list(range(len(structure_graph))))
    linker_atoms = all_atoms - node_atoms - floating_indices

    (
        linker_mol,
        linker_subgraph,
        linker_idx,
        linker_center,
        linker_coordinates,
    ) = get_bbs_from_indices(structure_graph, linker_atoms)

    node_mol, node_subgraph, node_idx, node_center, node_coordinates = get_bbs_from_indices(
        structure_graph, node_atoms
    )

    linker_atom_types = _linker_atom_types(linker_idx, node_idx, structure_graph)

    linker_atom_types["nodes"] = node_idx

    return linker_atom_types


def _linker_atom_types(indices, node_indices, structure_graph):
    """Group linker atoms in `connecting`, `functional_group` and `scaffold`
    (similar to how it has been done in molsimplify)."""
    functional_group = []
    scaffold = []
    connecting = []

    flat_node_indices = sum(node_indices, [])

    for index_group in indices:
        functional_group_ = []
        scaffold_ = []
        connecting_ = []
        for index in index_group:
            neighors = get_connected_site_indices(structure_graph, index)
            if any(i in flat_node_indices for i in neighors):
                connecting_.append(index)
            elif structure_graph.structure[index].specie.symbol not in ("H", "C"):
                functional_group_.append(index)
            else:
                scaffold_.append(index)
        functional_group.append(functional_group_)
        scaffold.append(scaffold_)
        connecting.append(connecting_)

    return {
        "linker_all": indices,
        "linker_functional": functional_group,
        "linker_scaffold": scaffold,
        "linker_connecting": connecting,
    }
