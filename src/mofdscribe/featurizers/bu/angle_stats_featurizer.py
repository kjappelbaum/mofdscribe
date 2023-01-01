# -*- coding: utf-8 -*-
"""Describe molecules by computing statistics for the distribution of angles A-COM-B between atoms A, B and COM.

COM is the center of mass of the molecule.
"""

from typing import List, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IMolecule, Molecule
from pymatgen.util.coord import get_angle

from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.extend import operates_on_imolecule, operates_on_molecule


@operates_on_molecule
@operates_on_imolecule
class PairWiseAngleStats(BaseFeaturizer):
    _NAME = "PairWiseAngleStats"

    def __init__(self, aggreations: Tuple[str] = ("mean", "std", "min", "max")) -> None:
        """Create a new PairwiseDistanceStats featurizer.

        Args:
            aggreations (Tuple[str]): Aggreations to compute.
                Defaults to ("mean", "std", "min", "max").
        """
        self.aggreations = aggreations

    def feature_labels(self) -> List[str]:
        return [f"{self._NAME}_{a}" for a in self.aggreations]

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

        feats = [ARRAY_AGGREGATORS[a](angles) for a in self.aggreations]

        return np.array(feats)

    def featurize(self, molecule: Union[Molecule, IMolecule]) -> np.ndarray:
        return self._featurize(molecule)

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
