"""Naive fragmentation implementation following Moosavi et al. 
which does not require multiple loops over the graph.
For alternative (slower) implementation see MOFfragmentor"""


from pymatgen.core import Structure
from pyymatgen.analysis.graphs import StructureGraph
from mofdscribe.utils.substructures import get_metal_indices
from mofdscribe.utils.structure_graph import get_connected_site_indices


def get_node_atoms(structure_graph: StructureGraph):
    metal_indices = get_metal_indices(structure_graph.structure)

    # make a set of all metals and atoms connected to them:
    node_set = set()
    for metal_index in metal_indices:
        node_set.add(metal_index)
        bonded_to_metal = get_connected_site_indices(structure_graph, metal_index)
        node_set.update(bonded_to_metal)

    # add atoms that are only connected to metal or Hydrogen to the node list + hydrogen atoms connected to them
    for atom in node_set:
        all_bonded_atoms = get_bonded_to_atom(self.adjacency_matrix, atom)
        only_bonded_metal_hydrogen = True
        for val in all_bonded_atoms:
            if not ((self.atom_types[val].upper() == "H") or val in (metal_list)):
                only_bonded_metal_hydrogen = False
        if only_bonded_metal_hydrogen:
            node_atom_set.update(set([atom]))

    final_node_atom_set = copy.deepcopy(node_atom_set)
    for atom in node_atom_set:
        for val in get_bonded_to_atom(self.adjacency_matrix, atom):
            if self.atom_types[val].upper() == "H":
                final_node_atom_set.update(set([val]))
    return final_node_atom_set
