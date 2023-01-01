# -*- coding: utf-8 -*-
"""Atom count featurizer."""
from typing import List, Union

import numpy as np
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)
from mofdscribe.mof import MOF


@operates_on_structure
@operates_on_istructure
@operates_on_molecule
@operates_on_imolecule
class NumAtoms(MOFBaseFeaturizer):
    """Featurizer that returns the number of atoms in a structure."""

    def __init__(self) -> None:
        """Construct a new NumAtoms featurizer."""
        super().__init__()

    def _featurize(
        self, structure_object: Union[Structure, IStructure, Molecule, IMolecule]
    ) -> np.ndarray:
        if not isinstance(structure_object, (Structure, IStructure, Molecule, IMolecule)):
            raise ValueError("Structure object must be a pymatgen Structure or Molecule.")
        return np.array([len(structure_object)])

    def featurize(self, mof: MOF) -> int:
        return len(mof.structure)

    def feature_labels(self) -> List[str]:
        return ["num_atoms"]

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
