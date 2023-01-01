# -*- coding: utf-8 -*-
"""perform analyses on structure graphs."""
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

import networkx as nx
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IStructure
from structuregraph_helpers.create import get_structure_graph as get_sg


def leads_to_terminal(nx_graph, edge, bridges: Dict[int, int] = None):
    if bridges is None:
        bridges = _generate_bridges(nx_graph)
    sorted_edge = sorted(edge)
    try:
        bridge_edge = bridges[sorted_edge[0]]
        return sorted_edge[1] in bridge_edge
    except KeyError:
        return False


def _generate_bridges(nx_graph) -> Dict[int, int]:

    bridges = list(nx.bridges(nx_graph))

    bridges_dict = defaultdict(list)
    for key, value in bridges:
        bridges_dict[key].append(value)

    return bridges_dict


def get_neighbors_up_to_scope(structure_graph, site_index, scope):
    """Get only the neighbors at a certain scope.

    That is, scope=3 will return all neighbors three bonds away from the
    site_index.

    Args:
        structure_graph (StructureGraph): The structure graph.
        site_index (int): The site index.
        scope (int): The scope.

    Returns:
        list: The list of neighbors.
    """
    neighbors_in_scope = defaultdict(set)
    neighbors_in_scope[0] = [site_index]
    visited = set([site_index])
    for i in range(1, scope + 1):
        for site in neighbors_in_scope[i - 1]:
            neighbors = get_connected_site_indices(structure_graph, site)
            neighbors = [n for n in neighbors if n not in visited]
            neighbors_in_scope[i].update(neighbors)
            visited.update(neighbors)

    return neighbors_in_scope


@lru_cache()
def get_structure_graph(structure: IStructure, strategy: Optional[str] = None) -> StructureGraph:
    """Get a StructureGraph object for a given structure.

    Args:
        structure (IStructure): Pymatgen structure object.
        strategy (str): Heuristic for assigning bonds.
            Must be one of 'jmolnn', 'crystalnn', 'isayevnn', 'vesta'. Defaults to None.

    Returns:
        StructureGraph: pymatgen StructureGraph object
    """
    return get_sg(structure, strategy)


def get_connected_site_indices(structure_graph: StructureGraph, site_index: int) -> List[int]:
    """Get all connected site indices for a given site."""
    connected_sites = structure_graph.get_connected_sites(site_index)
    return [site.index for site in connected_sites]
