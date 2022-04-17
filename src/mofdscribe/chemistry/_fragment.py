"""Naive fragmentation implementation following Moosavi et al. 
which does not require multiple loops over the graph.
For alternative (slower) implementation see MOFfragmentor"""


from copy import deepcopy
from typing import Set

from pymatgen.core import Structure
from pyymatgen.analysis.graphs import StructureGraph

from mofdscribe.utils.structure_graph import get_connected_site_indices, get_subgraphs_as_molecules
from mofdscribe.utils.substructures import _not_relevant_structure_indices, get_metal_indices


def get_node_atoms(structure_graph: StructureGraph) -> Set[int]:
    metal_indices = get_metal_indices(structure_graph.structure)
    metal_names = [str(structure_graph.structure[i].specie.symbol) for i in metal_indices]

    node_set = set()
    node_set.update(metal_indices)

    tmp_node_set = set()
    for metal_index in metal_indices:
        tmp_node_set.add(metal_index)
        bonded_to_metal = get_connected_site_indices(structure_graph, metal_index)
        tmp_node_set.update(bonded_to_metal)

    # add atoms that are only connected to metal or Hydrogen to the node list + hydrogen atoms connected to them
    for node_atom_index in tmp_node_set:
        all_bonded_atoms = get_connected_site_indices(structure_graph, node_atom_index)
        only_bonded_metal_hydrogen = True
        for index in all_bonded_atoms:
            if not ((str(structure_graph.structure[index].specie) == "H") or index in metal_names):
                only_bonded_metal_hydrogen = False
        if only_bonded_metal_hydrogen:
            node_set.add(index)

    final_node_atom_set = deepcopy(node_set)
    for atom_index in node_set:
        for index in get_connected_site_indices(structure_graph, atom_index):
            if str(structure_graph.structure[index].specie) == "H":
                final_node_atom_set.update(index)

    return final_node_atom_set


def get_floating_indices(structure_graph: StructureGraph) -> Set[int]:
    _, _, idx, _, _ = get_subgraphs_as_molecules(structure_graph)
    return set(idx)


def get_bbs_from_indices(structure_graph: StructureGraph, indices: Set[int]):
    graph_ = structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(graph_.structure, indices)
    graph_.remove_nodes(to_delete)
    mol, return_subgraphs, idx, centers, coordinates = get_subgraphs_as_molecules(graph_)
    return mol, return_subgraphs, idx, centers, coordinates


def fragment(structure_graph: StructureGraph):
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
