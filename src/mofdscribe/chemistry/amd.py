# -*- coding: utf-8 -*-
from typing import List, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.substructures import filter_element

__all__ = ["AMD"]


class AMD(BaseFeaturizer):
    """
    Implements the average minimum distance (AMD) isometry invariant.
    The AMD is the average of the point-wise distance distribution (PDD) of a crystal. The PDD lists distances to neighbouring atoms in order, closest first. Hence, the kth AMD value corresponds to the average distance to the kth nearest neighbour.

    The descriptors can be computed over the full structure or substructures of certain atom types.
    """

    def __init__(
        self,
        k: int = 100,
        atom_types=(
            "C-H-N-O",
            "F-Cl-Br-I",
            "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu",
        ),
        compute_for_all_elements: bool = True,
    ) -> None:
        """Initializes the AMD descriptor.

        Args:
            k (int, optional): controls the number of nearest neighbour atoms considered for each atom in the unit cell. Defaults to 100.
            atom_types (tuple, optional): Atoms that are used to create substructures for which the AMD descriptor is computed. Defaults to ( 'C-H-N-O', 'F-Cl-Br-I', 'Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu', ).
            compute_for_all_elements (bool, optional): If True, compute the AMD descriptor for the original structure with all elements. Defaults to True.
        """
        self.k = k
        atom_types = [] if atom_types is None else atom_types
        self.elements = atom_types
        self.atom_types = (
            list(atom_types) + ["all"] if compute_for_all_elements else list(atom_types)
        )
        self.compute_for_all_elements = compute_for_all_elements

    def _get_feature_labels(self) -> List[str]:
        labels = []
        for atom_type in self.atom_types:
            for i in range(self.k):
                labels.append(f"{atom_type}_{i}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        """
        Computes the AMD descriptor for a given structure.

        Args:
            structure: Structure to compute the descriptor for.

        Returns:
            A numpy array containing the AMD descriptor.
        """
        from amd import AMD as AMDBase
        from amd import PeriodicSet

        # ToDo: we can potentially parallelize this
        all_desc = []
        if len(self.elements) > 0:
            for element in self.elements:
                filtered_structure = filter_element(structure, element)
                all_desc.extend(
                    AMDBase(
                        PeriodicSet(
                            filtered_structure.cart_coords, filtered_structure.lattice.matrix
                        ),
                        self.k,
                    )
                )

        if self.compute_for_all_elements:
            all_desc.extend(
                AMDBase(
                    PeriodicSet(structure.cart_coords, structure.lattice.matrix),
                    self.k,
                )
            )

        return np.array(all_desc)

    def citations(self):
        return [
            "@article{amd2022,"
            "title = {Average Minimum Distances of periodic point sets - foundational invariants for mapping periodic crystals},"
            "author = {Daniel Widdowson and Marco M Mosca and Angeles Pulido and Vitaliy Kurlin and Andrew I Cooper},"
            "journal = {MATCH Communications in Mathematical and in Computer Chemistry},"
            "doi = {10.46793/match.87-3.529W},"
            "volume = {87},"
            "number = {3},"
            "pages = {529-559},"
            "year = {2022}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
