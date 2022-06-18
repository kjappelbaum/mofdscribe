# -*- coding: utf-8 -*-
"""Compute histograms of persistent images for MOFs."""
from typing import List, Optional, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from ._tda_helpers import get_diagrams_for_structure


class PHHist(BaseFeaturizer):
    """Featurizer that computes 2D histogram of persistent images.

    Compute a fixed-length vector of topological descriptors for a structure by
    summarizing the persistence diagrams of the structure (or substructure)
    usimg a 2D histogram.

    The descriptors can be computed over the full structure or substructures of
    certain atom types.
    """

    def __init__(
        self,
        atom_types: Optional[Tuple[str]] = (
            "C-H-N-O",
            "F-Cl-Br-I",
            "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V"
            "-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-"
            "Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-"
            "Th-Np-Lu-Rh-Pu",
        ),
        compute_for_all_elements: Optional[bool] = True,
        dimensions: Optional[Tuple[int]] = (1, 2),
        min_size: Optional[int] = 20,
        nx: Optional[int] = 10,
        ny: Optional[int] = 10,
        periodic: Optional[bool] = False,
        y_axis_label: Optional[str] = "persistence",
        normed: Optional[bool] = True,
    ) -> None:
        """Initialize the PHStats object.

        Args:
            atom_types (tuple, optional): Atoms that are used to create substructures
                for which the persistent homology statistics are computed.
                Defaults to ( "C-H-N-O", "F-Cl-Br-I",
                "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-
                Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-
                Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu", ).
            compute_for_all_elements (bool, optional): Compute descriptor for original structure with all atoms.
                Defaults to True.
            dimensions (Tuple[int], optional): Dimensions of topological features to consider.
                Defaults to (1, 2).
            min_size (int, optional): Minimum supercell size (in Angstrom).
                Defaults to 20.
            nx (int, optional): Number of points in the x-direction.
                Defaults to 10.
            ny (int, optional): Number of points in the y-direction.
                Defaults to 10.
            periodic (bool, optional): If true, then periodic Euclidean is used in the analysis (experimental!).
                Defaults to False.
            y_axis_label (str, optional): Label for the y-axis. Can be "death" or "persistence".
                Defaults to "persistence".
            normed (bool, optional): If true, then the histogram is normalized.
                Defaults to True.
        """
        atom_types = [] if atom_types is None else atom_types
        self.elements = atom_types
        self.atom_types = (
            list(atom_types) + ["all"] if compute_for_all_elements else list(atom_types)
        )
        self.compute_for_all_elements = compute_for_all_elements
        self.dimensions = dimensions
        self.min_size = min_size
        self.nx = nx
        self.ny = ny
        self.periodic = periodic
        self.y_axis_label = y_axis_label
        self.normed = normed

    def _get_feature_labels(self) -> List[str]:
        labels = []
        for atom_type in self.atom_types:
            for dim in self.dimensions:
                for nx in range(self.nx):
                    for ny in range(self.ny):
                        labels.append(f"{atom_type}_dim{dim}_nx{nx}_ny{ny}_{self.y_axis_label}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        res = get_diagrams_for_structure(
            structure,
            self.elements,
            self.compute_for_all_elements,
            self.min_size,
            periodic=self.periodic,
        )

        flat_results = []
        for atom_type in self.atom_types:
            for dim in self.dimensions:

                dimname = f"dim{dim}"

                diagram = res[atom_type][dimname]

                try:
                    d = np.array(
                        [[x["birth"], x["death"], x["death"] - x["birth"]] for x in diagram]
                    )
                except IndexError:
                    d = np.array([[x[0], x[1], x[1] - x[0]] for x in diagram])
                d = np.ma.masked_invalid(d)

                x = d[:, 0]
                y = d[:, 1] if self.y_axis_label == "death" else d[:, 2]

                hist = np.histogram2d(x, y, bins=(self.nx, self.ny), density=self.normed)[0]
                flat_results.append(hist.reshape(1, -1))
        return np.hstack(flat_results).flatten()

    def citations(self):
        return [
            "@article{doi:10.1021/acs.jpcc.0c01167,"
            "author = {Krishnapriyan, Aditi S. and Haranczyk, Maciej and Morozov, Dmitriy},"
            "title = {Topological Descriptors Help Predict Guest Adsorption in Nanoporous Materials},"
            "journal = {The Journal of Physical Chemistry C},"
            "volume = {124},"
            "number = {17},"
            "pages = {9360-9368},"
            "year = {2020},"
            "doi = {10.1021/acs.jpcc.0c01167},"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
