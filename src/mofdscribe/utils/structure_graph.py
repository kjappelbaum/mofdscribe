# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import lru_cache
from typing import List, Set, Tuple

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import CrystalNN, IsayevNN, JmolNN
from pymatgen.core import IStructure, Molecule, Structure


def get_neighbors_at_distance(structure_graph: StructureGraph, start: int, scope: int) -> Set[int]:
    # Todo: This code is stupid.
    neighbors_at_last_level = [start]
    all_neighbors = set()
    neighbors_at_next_level = []
    # print(structure_graph.get_connected_sites(start))
    for _ in range(scope):
        for n in neighbors_at_last_level:
            neighbors_at_next_level.extend(get_connected_site_indices(structure_graph, n))

        all_neighbors.update(neighbors_at_last_level)
        neighbors_at_last_level = neighbors_at_next_level
        neighbors_at_next_level = []
    all_neighbors.remove(start)
    neighbors_at_last_level = set(neighbors_at_last_level)
    if start in neighbors_at_last_level:
        neighbors_at_last_level.remove(start)

    return all_neighbors, neighbors_at_last_level


def _get_local_env_strategy(name: str = None):
    n = "jmolnn" if name is None else name.lower()

    if n == "jmolnn":
        return JmolNN()
    elif n == "crystalnn":
        return CrystalNN()
    elif n == "isayevnn":
        return IsayevNN()


@lru_cache()
def get_structure_graph(structure: IStructure, strategy: str = None) -> StructureGraph:
    strategy = _get_local_env_strategy(strategy)
    sg = StructureGraph.with_local_env_strategy(structure, strategy)
    nx.set_node_attributes(
        sg.graph,
        name="idx",
        values=dict(zip(range(len(sg)), range(len(sg)))),
    )
    return sg


def get_connected_site_indices(structure_graph: StructureGraph, site_index: int) -> List[int]:
    connected_sites = structure_graph.get_connected_sites(site_index)
    return [site.index for site in connected_sites]


def _is_in_cell(frac_coords):
    return all(frac_coords <= 1)


def _is_any_atom_in_cell(frac_coords):
    for row in frac_coords:
        if _is_in_cell(row):
            return True
    return False


def _select_parts_in_cell(  # pylint:disable=too-many-arguments, too-many-locals
    molecules: List[Molecule],
    graphs: List[MoleculeGraph],
    indices: List[List[int]],
    indices_here: List[List[int]],
    centers: List[np.ndarray],
    fractional_coordinates: np.ndarray,
    coordinates: np.ndarray,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]]]:
    valid_indices = defaultdict(list)
    for i, ind in enumerate(indices_here):
        # change this check to having an atom in the cell
        frac_coords = fractional_coordinates[ind]

        if _is_any_atom_in_cell(frac_coords):
            sorted_idx = sorted(indices[i])
            valid_indices[str(sorted_idx)].append(i)

    molecules_ = []
    selected_indices = []
    graphs_ = []
    centers_ = []
    coordinates_ = []

    for _, values in valid_indices.items():
        for index in values:
            selected_indices.append(indices[index])
            molecules_.append(molecules[index])
            graphs_.append(graphs[index])
            centers_.append(centers[index])
            coordinates_.append(coordinates[index])

    return molecules_, graphs_, selected_indices, centers_, coordinates_


def get_subgraphs_as_molecules(  # pylint:disable=too-many-locals
    structure_graph: StructureGraph,
    use_weights: bool = False,
    return_unique: bool = True,
    disable_boundary_crossing_check: bool = False,
    filter_in_cell: bool = True,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]], List[np.ndarray]]:
    """Copied from
    http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    and removed the duplicate check
    Args:
        structure_graph ( pymatgen.analysis.graphs.StructureGraph): Structuregraph
        use_weights (bool): If True, use weights for the edge matching
        return_unique (bool): If true, it only returns the unique molecules.
            If False, it will return all molecules that are completely
            included in the unit cell
            and fragments of the ones that are only partly in the cell
        filter_in_cell (bool): If True, it will only return molecules that
            have at least one atom in the cell
    Returns:
        Tuple[List[Molecule], List[MoleculeGraph], List[List[int]], List[np.ndarray]]
    """
    # pylint: disable=invalid-name
    # creating a supercell is an easy way to extract
    # molecules (and not, e.g., layers of a 2D crystal)
    # without adding extra logic
    supercell_sg = structure_graph * (3, 3, 3)

    # make undirected to find connected subgraphs
    supercell_sg.graph = nx.Graph(supercell_sg.graph)

    # find subgraphs
    all_subgraphs = [
        supercell_sg.graph.subgraph(c).copy() for c in nx.connected_components(supercell_sg.graph)
    ]

    # discount subgraphs that lie across *supercell* boundaries
    # these will subgraphs representing crystals
    molecule_subgraphs = []

    for subgraph in all_subgraphs:
        if disable_boundary_crossing_check:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))
        else:
            intersects_boundary = any(  # pylint: disable=use-a-generator
                [d["to_jimage"] != (0, 0, 0) for u, v, d in subgraph.edges(data=True)]
            )
            if not intersects_boundary:
                molecule_subgraphs.append(nx.MultiDiGraph(subgraph))

    # add specie names to graph to be able to test for isomorphism
    for subgraph in molecule_subgraphs:
        for node in subgraph:
            subgraph.add_node(node, specie=str(supercell_sg.structure[node].specie))

    unique_subgraphs = []

    def node_match(node_1, node_2):
        return node_1["specie"] == node_2["specie"]

    def edge_match(edge_1, edge_2):
        if use_weights:
            return edge_1["weight"] == edge_2["weight"]

        return True

    if return_unique:
        for subgraph in molecule_subgraphs:
            already_present = [
                nx.is_isomorphic(subgraph, g, node_match=node_match, edge_match=edge_match)
                for g in unique_subgraphs
            ]

            if not any(already_present):
                unique_subgraphs.append(subgraph)

    def make_mols(
        molecule_subgraphs=molecule_subgraphs, center=False
    ):  # pylint:disable=dangerous-default-value
        molecules = []
        indices = []
        indices_here = []
        mol_centers = []
        coordinates = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[node].coords for node in subgraph.nodes()]
            species = [supercell_sg.structure[node].specie for node in subgraph.nodes()]

            # binding = [
            #     supercell_sg.structure[n].properties["binding"]
            #     for n in subgraph.nodes()
            # ]
            idx = [subgraph.nodes[node]["idx"] for node in subgraph.nodes()]
            idx_here = subgraph.nodes()
            molecule = Molecule(species, coords)  #  site_properties={"binding": binding}
            mol_centers.append(np.mean(supercell_sg.structure.cart_coords[idx_here], axis=0))
            # shift so origin is at center of mass
            if center:
                molecule = molecule.get_centered_molecule()
            indices.append(idx)
            molecules.append(molecule)
            indices_here.append(idx_here)
            coordinates.append(coords)
        return molecules, indices, indices_here, mol_centers, coordinates

    def relabel_graph(multigraph):
        mapping = dict(zip(multigraph, range(0, len(multigraph.nodes()))))
        return nx.readwrite.json_graph.adjacency_data(nx.relabel_nodes(multigraph, mapping))

    if return_unique:
        mol, idx, indices_here, centers, coordinates = make_mols(unique_subgraphs, center=True)
        return_subgraphs = unique_subgraphs
        return (
            mol,
            [MoleculeGraph(mol, relabel_graph(graph)) for mol, graph in zip(mol, return_subgraphs)],
            idx,
            centers,
            coordinates,
        )

    mol, idx, indices_here, centers, coordinates = make_mols(molecule_subgraphs)

    return_subgraphs = [
        MoleculeGraph(mol, relabel_graph(graph)) for mol, graph in zip(mol, molecule_subgraphs)
    ]

    if filter_in_cell:
        mol, return_subgraphs, idx, centers, coordinates = _select_parts_in_cell(
            mol,
            return_subgraphs,
            idx,
            indices_here,
            centers,
            structure_graph.structure.lattice.get_fractional_coords(
                supercell_sg.structure.cart_coords
            ),
            coordinates,
        )

    return mol, return_subgraphs, idx, centers, coordinates
