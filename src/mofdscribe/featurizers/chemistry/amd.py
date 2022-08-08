# -*- coding: utf-8 -*-
"""Generalized average minimum distance (AMD) featurizer."""
from typing import List, Tuple, Union

import numpy as np
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.extend import operates_on_istructure, operates_on_structure
from mofdscribe.featurizers.utils.substructures import filter_element

__all__ = ["AMD"]


@operates_on_istructure
@operates_on_structure
class AMD(MOFBaseFeaturizer):
    """Implements the generalized average minimum distance (AMD) isometry invariant [Widdowson2022]_.

    Note that it currently does not implement averages according to
    multiplicity of sites (as the original code supports).
    The generalization is to other aggregations of the PDD.

    The AMD is the average of the point-wise distance distribution (PDD) of a
    crystal. The PDD lists distances to neighbouring atoms in order, closest
    first. Hence, the kth AMD value corresponds to the average distance to the
    kth nearest neighbour.

    The descriptors can be computed over the full structure or substructures of
    certain atom types.
    """

    def __init__(
        self,
        k: int = 100,
        atom_types: Tuple[str] = (
            "C-H-N-O",
            "F-Cl-Br-I",
            "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V"
            "-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-"
            "Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-"
            "Th-Np-Lu-Rh-Pu",
        ),
        compute_for_all_elements: bool = True,
        aggregations: Tuple[str] = ("mean",),
        primitive: bool = True,
    ) -> None:
        """Initialize the AMD descriptor.

        Args:
            k (int): controls the number of nearest neighbour atoms considered
                for each atom in the unit cell. Defaults to 100.
            atom_types (tuple): Atoms that are used to create substructures
                for which the AMD descriptor is computed.
                Defaults to ( 'C-H-N-O', 'F-Cl-Br-I',
                'Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-
                Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-
                Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu', ).
            compute_for_all_elements (bool): If True, compute the AMD descriptor for
                the original structure with all elements. Defaults to True.
            aggregations (tuple): Aggregations of the AMD descriptor.
                The 'mean' is equivalent to the original AMD. Defaults to ('mean',).
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to True.
        """
        self.k = k
        atom_types = [] if atom_types is None else atom_types
        self.elements = atom_types
        self.atom_types = (
            list(atom_types) + ["all"] if compute_for_all_elements else list(atom_types)
        )
        self.compute_for_all_elements = compute_for_all_elements
        self.aggregations = aggregations
        super().__init__(primitive=primitive)

    def _get_feature_labels(self) -> List[str]:
        labels = []
        for atom_type in self.atom_types:
            for agg in self.aggregations:
                for i in range(self.k):
                    labels.append(f"amd_{atom_type}_{agg}_{i}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def _featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        """Compute the AMD descriptor for a given structure.

        Args:
            structure (Union[Structure, IStructure]): Structure to compute the descriptor for.

        Returns:
            A numpy array containing the AMD descriptor.
        """
        from amd._nns import nearest_neighbours
        from amd.calculate import _extract_motif_cell
        from amd.periodicset import PeriodicSet

        def get_pdd(structure, k):
            motif, cell, asymmetric_unit, multiplicities = _extract_motif_cell(
                PeriodicSet(structure.cart_coords, structure.lattice.matrix)
            )
            pdd, _, _ = nearest_neighbours(motif, cell, asymmetric_unit, k)
            return pdd

        # ToDo: we can potentially parallelize this
        all_desc = []
        if len(self.elements) > 0:
            for element in self.elements:
                filtered_structure = filter_element(structure, element)
                if filtered_structure is not None:
                    pdd = get_pdd(filtered_structure, self.k)
                else:
                    pdd = np.empty((1, self.k))
                    pdd[:] = np.nan
                for agg in self.aggregations:
                    all_desc.append(getattr(np, agg)(pdd, axis=0))

        if self.compute_for_all_elements:
            pdd = get_pdd(structure, self.k)
            for agg in self.aggregations:
                all_desc.append(getattr(np, agg)(pdd, axis=0))

        return np.concatenate(all_desc)

    def citations(self):
        return [
            "@article{amd2022,"
            "title = {Average Minimum Distances of periodic point sets - "
            "foundational invariants for mapping periodic crystals},"
            "author = {Daniel Widdowson and Marco M Mosca and Angeles Pulido "
            "and Vitaliy Kurlin and Andrew I Cooper},"
            "journal = {MATCH Communications in Mathematical and in Computer Chemistry},"
            "doi = {10.46793/match.87-3.529W},"
            "volume = {87},"
            "number = {3},"
            "pages = {529-559},"
            "year = {2022}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka", "Daniel Widdowson"]
