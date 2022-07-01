# -*- coding: utf-8 -*-
"""Compute local structure order parameters for a fragment."""
from typing import List, Optional, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.local_env import LocalStructOrderParams
from pymatgen.core import IMolecule, IStructure, Molecule, Structure
from pymatgen.core.periodic_table import DummySpecies

from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)

_default_types = (
    "cn",
    "tet",
    "oct",
    "bcc",
    "sq_pyr",
    "sq_pyr_legacy",
    "tri_bipyr",
    "sq_bipyr",
    "oct_legacy",
    "tri_plan",
    "sq_plan",
    "pent_plan",
    "tri_pyr",
    "pent_pyr",
    "hex_pyr",
    "pent_bipyr",
    "hex_bipyr",
    "T",
    "cuboct",
    "oct_max",
    "tet_max",
    "tri_plan_max",
    "sq_plan_max",
    "pent_plan_max",
    "cuboct_max",
    "bent",
    "see_saw_rect",
    "hex_plan_max",
    "sq_face_cap_trig_pris",
)


@operates_on_structure
@operates_on_istructure
@operates_on_molecule
@operates_on_imolecule
class LSOP(BaseFeaturizer):
    """Compute shape parameters for a fragment.

    The fragments can be a molecule or a molecule that only contains important part (e.g. binding sites)
    of a molecule. The shape parameters are then supposed to quantify the shape of the fragment.
    For instance, a triangular molecule will have a `tri_plan` parameter close to 1.

    While there is a site-based LSOP featurizer in matminer there is none that uses LSOP
    to quantify the shape of some fragment. This featurizers just does that.

    It does so by placing a dummy site at the center of mass
    of the fragment and then computes the LSOP considering all other sites as neighbors.
    """

    def __init__(
        self, types: Tuple[str] = _default_types, parameters: Optional[List[dict]] = None
    ) -> None:
        """Initialize the featurizer.

        Args:
            types (Tuple[str]): The types of LSOP to compute.
                For the full list of types
                see: :py:attr:`pymatgen.analysis.local_env.LocalStructOrderParams. __supported_types`.
                Defaults to: ["tet", "oct", "bcc", "sq_pyr", "sq_pyr_legacy", "tri_bipyr", "sq_bipyr",
                "oct_legacy", "tri_plan", "sq_plan", "pent_plan", "tri_pyr", "pent_pyr", "hex_pyr",
                "pent_bipyr", "hex_bipyr", "T", "cuboct", "oct_max", "tet_max", "tri_plan_max",
                "sq_plan_max", "pent_plan_max", "cuboct_max", "bent", "see_saw_rect", "hex_plan_max",
                "sq_face_cap_trig_pris"]
            parameters(List[dict], optional): The parameters to pass to the LocalStructOrderParams object.
        """
        self._lsop = LocalStructOrderParams(types, parameters)
        self.types = types

    def feature_labels(self) -> List[str]:
        return [f"lsop_{val}" for val in self.types]

    def featurize(self, s: Union[Structure, IStructure, Molecule, IMolecule]) -> np.ndarray:
        molecule = Molecule.from_sites(s.sites)
        com = molecule.center_of_mass
        orginal_len = len(molecule)
        molecule.append(DummySpecies(), com)
        lsop = self._lsop.get_order_parameters(
            molecule, len(molecule) - 1, indices_neighs=np.arange(orginal_len)
        )
        lsop = [f if f is not None else np.nan for f in lsop]
        return np.array(lsop)

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return [
            "@article{Zimmermann2020,"
            "doi = {10.1039/c9ra07755c},"
            "url = {https://doi.org/10.1039/c9ra07755c},"
            "year = {2020},"
            "publisher = {Royal Society of Chemistry ({RSC})},"
            "volume = {10},"
            "number = {10},"
            "pages = {6063--6081},"
            "author = {Nils E. R. Zimmermann and Anubhav Jain},"
            "title = {Local structure order parameters and site fingerprints"
            " for quantification of coordination environment and crystal structure similarity},"
            "journal = {{RSC} Advances}"
            "}"
        ]
