# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import Structure


class PHImage(BaseFeaturizer):
    def __init__(self, atom_types=Tuple[str]) -> None:
        ...

    def feature_labels(self) -> List[str]:
        return ...

    def featurize(self, structure: Structure) -> np.ndarray:
        return []

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
