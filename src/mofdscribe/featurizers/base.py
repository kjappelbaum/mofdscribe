from typing import Union

import numpy as np
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import BaseFeaturizer


class MOFBaseFeaturizer(BaseFeaturizer):
    def __init__(self, primitive: bool = True) -> None:
        self.primitive = primitive

    def _get_primitive(self, structure: Union[Structure, IStructure]) -> Structure:
        return structure.get_primitive_structure()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        if self.primitive:
            structure = self._get_primitive(structure)
        return self._featurize(structure)
