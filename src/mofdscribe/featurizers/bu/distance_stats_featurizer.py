# -*- coding: utf-8 -*-
"""Describe molecules by computing statistics of pairwise distances between their atoms."""
from typing import List, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)


@operates_on_molecule
@operates_on_imolecule
@operates_on_istructure
@operates_on_structure
class PairwiseDistanceStats(BaseFeaturizer):
    """
    Describe the shape of molecules by computing statistics of pairwise distances.

    For doing so, we will just compute all pairwise distances and then compute
    some statistics on them.
    One might also think of this as pretty rough approximation of something like
    the AMD fingerpint.
    """

    def __init__(self, aggregations: Tuple[str] = ("mean", "std", "max", "min")) -> None:
        """Create a new PairwiseDistanceStats featurizer.

        Args:
            aggregations (Tuple[str], optional): Aggregations to compute over the pairwise
                distances. Must be one of :py:obj:`ARRAY_AGGREGATORS`.
                Defaults to ("mean", "std", "max", "min").
        """
        self.aggregations = aggregations

    def feature_labels(self) -> List[str]:
        return [f"pairwise_distance_stats_{a}" for a in self.aggregations]

    def featurize(self, structure: Union[Molecule, IMolecule, Structure, IStructure]) -> np.ndarray:
        distances = []
        for i, _ in enumerate(structure):
            for j, _ in enumerate(structure):
                if i < j:
                    distances.append(structure.get_distance(i, j))

        features = []
        for agg in self.aggregations:
            features.append(ARRAY_AGGREGATORS[agg](distances))
        return np.array(features)

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return []
