# -*- coding: utf-8 -*-
"""Describe molecules by computing histogram of angles A-COM-B between atoms A, B and COM.

COM is the center of mass of the molecule.
"""
from typing import List, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IMolecule, Molecule
from pymatgen.util.coord import get_angle

from mofdscribe.featurizers.utils.extend import operates_on_imolecule, operates_on_molecule
from mofdscribe.featurizers.utils.histogram import get_rdf
from mofdscribe.featurizers.utils.mixins import GetGridMixin


@operates_on_molecule
@operates_on_imolecule
class PairWiseAngleHist(BaseFeaturizer, GetGridMixin):
    _NAME = "PairWiseAngleHist"

    def __init__(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 180.0,
        bin_size: float = 20,
        density: bool = True,
    ) -> None:
        """Create a new PairwiseDistanceHist featurizer.

        Args:
            lower_bound (float): Lower bound of the histogram.
                Defaults to 0.0.
            upper_bound (float): Upper bound of the histogram.
                Defaults to 180.
            bin_size (float): Size of the bins.
                Defaults to 20.
            density (bool): Whether to return the density or the counts.
                Defaults to True.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_size = bin_size
        self.density = density

    def feature_labels(self) -> List[str]:
        return [
            f"{self._NAME}_{a}"
            for a in self._get_grid(self.lower_bound, self.upper_bound, self.bin_size)
        ]

    def _featurize(self, molecule: Union[Molecule, IMolecule]) -> np.ndarray:
        # Get the pairwise distances
        com = molecule.center_of_mass
        angles = []
        for i in range(len(molecule)):
            for j in range(len(molecule)):
                if i > j:
                    v1 = molecule.cart_coords[i] - com
                    v2 = molecule.cart_coords[j] - com
                    angles.append(get_angle(v1, v2))

        # Compute the histogram
        features = get_rdf(
            angles,
            lower_lim=self.lower_bound,
            upper_lim=self.upper_bound,
            bin_size=self.bin_size,
            density=self.density,
            normalized=False,
        )
        return features

    def featurize(self, molecule: Union[Molecule, IMolecule]) -> np.ndarray:
        return self._featurize(molecule)

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
