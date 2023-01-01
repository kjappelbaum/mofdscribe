# -*- coding: utf-8 -*-
"""Featurizer that computes statistics on the number of hops between binding/branching sites."""

from typing import Collection, Tuple, Union

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph

from mofdscribe.featurizers.graph.graphfeaturizer import GraphFeaturizer
from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.extend import (
    operates_on_moleculegraph,
    operates_on_structuregraph,
)
from mofdscribe.mof import MOF


def _hop_stats(
    structuregraph: Union[StructureGraph, MoleculeGraph], indices: Collection[int], aggregations
) -> np.ndarray:
    # for all combinations of indices compute the shortest path
    # and return the statistics of the shortest paths
    shortest_paths = []
    undirected = structuregraph.graph.to_undirected()
    for i_count, i in enumerate(indices):
        for j_count, j in enumerate(indices):
            if i_count > j_count:
                shortest_paths.append(nx.shortest_path_length(undirected, i, j))
    return np.array([ARRAY_AGGREGATORS[agg](shortest_paths) for agg in aggregations])


@operates_on_structuregraph
@operates_on_moleculegraph
class NumHops(GraphFeaturizer):
    """Featurizer that computes statistics on the shortest path lengths between sites."""

    _NAME = "num_hops"

    def __init__(self, aggregations: Tuple[str] = ("mean", "std", "min", "max")) -> None:
        """Construct a NumHops featurizer.

        Args:
            aggregations: The aggregations to compute on the shortest path lengths.
                Defaults to ("mean", "std", "min", "max").
        """
        self.aggregations = aggregations

    def _featurize(
        self, structuregraph: Union[StructureGraph, MoleculeGraph], indices: Collection[int]
    ) -> np.ndarray:
        # for all combinations of indices compute the shortest path
        # and return the statistics of the shortest paths
        return _hop_stats(structuregraph, indices, self.aggregations)

    def featurize(self, mof: MOF, indices: Collection[int]) -> np.ndarray:
        return self._featurize(mof.structure_graph, indices)

    def feature_labels(self) -> Collection[str]:
        return [f"{self._NAME}_{agg}" for agg in self.aggregations]

    def citations(self) -> Collection[str]:
        return []

    def implementors(self) -> Collection[str]:
        return ["Kevin Maik Jablonka"]
