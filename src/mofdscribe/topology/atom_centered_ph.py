# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np
from matminer.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure


# Todo: allow doing this with cutoff and coordination shells
# let's implement this as site-based featurizer for now
class AtomCenteredPH(BaseFeaturizer):
    def __init__(
        self,
        aggregation_types: Tuple[str],
        aggregation_functions: Tuple[str],
        cutoff: float = 12,
        symprec: float = 1e-1,
        angle_tolerance: float = 5,
    ) -> None:
        super().__init__()

    def featurize(self, s: Union[Structure, IStructure], idx: int) -> np.ndarray:
        ...

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return [
            "@article{Jiang2021,"
            "doi = {10.1038/s41524-021-00493-w},"
            "url = {https://doi.org/10.1038/s41524-021-00493-w},"
            "year = {2021},"
            "month = feb,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {7},"
            "number = {1},"
            "author = {Yi Jiang and Dong Chen and Xin Chen and Tangyi Li and Guo-Wei Wei and Feng Pan},"
            "title = {Topological representations of crystalline compounds for the machine-learning prediction of materials properties},"
            "journal = {npj Computational Materials}"
            "}"
        ]
