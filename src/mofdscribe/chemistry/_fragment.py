"""Naive fragmentation implementation following Moosavi et al. 
which does not require multiple loops over the graph.
For alternative (slower) implementation see MOFfragmentor"""


from pymatgen.core import Structure
from pyymatgen.analysis.graphs import StructureGraph


def get_node_atoms(structure_graph: StructureGraph):
    metal_list = set(self.metal_atoms)
    node_atom_set = set(metal_list)
    # make a set of all metals and atoms connected to them:
    tmpset = set()
    for metal_atom in metal_list:
        tmpset.add(metal_atom)
        bonded_to_metal = get_bonded_to_atom(self.adjacency_matrix, metal_atom)
        tmpset.update(bonded_to_metal)

    # add atoms that are only connected to metal or Hydrogen to the node list + hydrogen atoms connected to them
    for atom in tmpset:
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
