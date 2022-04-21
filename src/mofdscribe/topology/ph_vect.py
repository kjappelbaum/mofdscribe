# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import List, Tuple

from matminer.featurizers.base import BaseFeaturizer

# ToDo: implement https://github.com/scikit-tda/pervect


class PHVect(BaseFeaturizer):
    """

    Computes persistent homology barcodes.
    Typically, persistent barcodes are computed for all atoms in the structure.
    However, one can also compute persistent barcodes for a subset of atoms.
    This can be done by specifying the atom types in the constructor.
    """

    def __init__(self, atom_types=Tuple[str]) -> None:
        self.transformers = defaultdict(lambda: defaultdict(dict))

    def feature_labels(self) -> List[str]:
        return ...

    def citations(self):
        return []

    def implementors(self):
        return ["Kevin Maik Jablonka"]
