from pymatgen.core import IStructure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JMolNN, CrystalNN, IsayevNN
from functools import lru_cache
from typing import List


def _get_local_env_strategy(name: str):
    n = name.lower()
    if n == "jmolnn":
        return JMolNN()
    elif n == "crystalnn":
        return CrystalNN()
    elif n == "isayevnn":
        return IsayevNN()


@lru_cache()
def get_structure_graph(structure: IStructure, strategy: str) -> StructureGraph:
    strategy = _get_local_env_strategy(strategy)
    sg = StructureGraph.with_local_env_strategy(structure, strategy)
    return sg


def get_connected_site_indices(structure_graph: StructureGraph, site_index: int) -> List[int]:
    connected_sites = structure_graph.get_connected_sites(site_index)
    return [site.index for site in connected_sites]
