"""Naive fragmentation implementation following Moosavi et al. 
which does not require multiple loops over the graph.
For alternative (slower) implementation see MOFfragmentor"""


from copy import deepcopy
from typing import Set

from pymatgen.core import Structure
from pyymatgen.analysis.graphs import StructureGraph

from mofdscribe.utils.structure_graph import get_connected_site_indices
from mofdscribe.utils.substructures import get_metal_indices


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
    pass
