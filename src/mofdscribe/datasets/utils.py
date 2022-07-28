"""Cached graphs."""

from functools import lru_cache
from structuregraph_helpers.create import get_structure_graph
from pymatgen.core import IStructure
from pymatgen.analysis.graphs import StructureGraph

from structuregraph_helpers.hash import (
    decorated_graph_hash,
    undecorated_graph_hash,
    decorated_scaffold_hash,
    undecorated_scaffold_hash,
)


@lru_cache(maxsize=None)
def get_structure_graph_cached(structure: IStructure) -> StructureGraph:
    return get_structure_graph(structure)


@lru_cache(maxsize=None)
def get_decorated_graph_hash_cached(structure: IStructure) -> str:
    return decorated_graph_hash(get_structure_graph_cached(structure))


@lru_cache(maxsize=None)
def get_undecorated_graph_hash_cached(structure: IStructure) -> str:
    return undecorated_graph_hash(get_structure_graph_cached(structure))


@lru_cache(maxsize=None)
def get_decorated_scaffold_hash_cached(structure: IStructure) -> str:
    return decorated_scaffold_hash(get_structure_graph_cached(structure))


@lru_cache(maxsize=None)
def get_undecorated_scaffold_hash_cached(structure: IStructure) -> str:
    return undecorated_scaffold_hash(get_structure_graph_cached(structure))
