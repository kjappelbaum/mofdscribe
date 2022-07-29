# -*- coding: utf-8 -*-
"""Describe molecules by computing a histogram of pairwise distances between their atoms."""
from typing import List, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)
from mofdscribe.featurizers.utils.histogram import get_rdf


@operates_on_molecule
@operates_on_imolecule
@operates_on_istructure
@operates_on_structure
class PairwiseDistanceHist(BaseFeaturizer):
    """
    Describe the shape of molecules by computing a histogram of pairwise distances.

    For doing so, we will just compute all pairwise distances and then compute
    the histogram of them.
    One might also think of this as pretty rough approximation of something like
    the AMD fingerpint

    It also has some similarities to the "Grouped representation of interatomic distances"
    reported in [Zhang2022]_.
    """

    def __init__(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 15.0,
        bin_size: float = 0.5,
        density: bool = True,
    ) -> None:
        """Create a new PairwiseDistanceHist featurizer.

        Args:
            lower_bound (float): Lower bound of the histogram.
                Defaults to 0.0.
            upper_bound (float): Upper bound of the histogram.
                Defaults to 15.0.
            bin_size (float): Size of the bins.
                Defaults to 0.5.
            density (bool): Whether to return the density or the counts.
                Defaults to True.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_size = bin_size
        self.density = density

    def _get_grid(self):
        return np.arange(self.lower_bound, self.upper_bound, self.bin_size)

    def feature_labels(self) -> List[str]:
        return [f"pairwise_distance_hist_{a}" for a in self._get_grid()]

    def featurize(self, structure: Union[Molecule, IMolecule, Structure, IStructure]) -> np.ndarray:
        distances = []
        for i, _ in enumerate(structure):
            for j, _ in enumerate(structure):
                if i < j:
                    distances.append(structure.get_distance(i, j))

        features = get_rdf(
            distances,
            lower_lim=self.lower_bound,
            upper_lim=self.upper_bound,
            bin_size=self.bin_size,
            density=self.density,
            normalized=False,
        )
        return features

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return []
