# -*- coding: utf-8 -*-
"""perform analyses on structure graphs."""
from functools import lru_cache
from typing import List, Optional, Set, Tuple

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IStructure
from structuregraph_helpers.create import get_structure_graph as get_sg


def get_neighbors_at_distance(
    structure_graph: StructureGraph, start: int, scope: int
) -> Tuple[Set[int], List[int]]:
    """For a structure graph and a start site, return all sites within a certain\
        distance (scope) of the start site.

    Args:
        structure_graph (StructureGraph): pymatgen StructureGraph object
        start (int): starting atom
        scope (int): distance to search

    Returns:
        Tuple[Set[int], List[int]]: All sites within the scope of the start
        site, and the indices of the sites in the last shell
    """
    # Todo: This code is stupid.
    neighbors_at_last_level = [start]
    all_neighbors = set()
    neighbors_at_next_level = []
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
