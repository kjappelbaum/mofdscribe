# -*- coding: utf-8 -*-
from typing import List, Tuple

from matminer.featurizers.base import BaseFeaturizer


class PHBarcode(BaseFeaturizer):
    """
    Computes persistent homology barcodes.
    The implemention here also allows to do this for sets of atom types.
    This approach is novel and allows to incorporate chemical information in the featurization
    using persistent homology.


    """

    def __init__(self, atom_types=Tuple[str]) -> None:
        ...

    def feature_labels(self) -> List[str]:
        return ...

    def citations(self):
        return []

    def implementors(self):
        return ["Kevin Maik Jablonka"]
