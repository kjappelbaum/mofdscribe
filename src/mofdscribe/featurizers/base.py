"""Base featurizer for MOF structure based featurizers."""
from typing import Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure


class MOFBaseFeaturizer(BaseFeaturizer):
    """Base featurizer for MOF structure based featurizers."""

    def __init__(self, primitive: bool = True) -> None:
        """
        Args:
            primitive (bool): If True, use the primitive cell of the structure.
                Defaults to True.
        """
        self.primitive = primitive

    def _get_primitive(self, structure: Union[Structure, IStructure]) -> Structure:
        return structure.get_primitive_structure()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        """Compute the descriptor for a given structure.

        Args:
            structure (Union[Structure, IStructure]): Structure to compute the descriptor for.

        Returns:
            A numpy array containing the descriptor.
        """
        if self.primitive:
            structure = self._get_primitive(structure)
        return self._featurize(structure)
