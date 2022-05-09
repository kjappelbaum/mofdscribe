# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.utils import flatten

from ._tda_helpers import get_diagrams_for_structure, persistent_diagram_stats


class PHStats(BaseFeaturizer):
    """
    Compute a fixed-length vector of topological descriptors for a structure by summarizing the persistence diagrams
    of the structure (or substructure) using aggegrations such as `min`, `max`, `mean`, and `std`.

    The descriptors can be computed over the full structure or substructures of certain atom types.
    """

    def __init__(
        self,
        atom_types=(
            "C-H-N-O",
            "F-Cl-Br-I",
            "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu",
        ),
        compute_for_all_elements: bool = True,
        dimensions: Tuple[int] = (1, 2),
        min_size: int = 20,
        aggregation_functions: Tuple[str] = ("min", "max", "mean", "std"),
    ) -> None:
        """_summary_

        Args:
            atom_types (tuple, optional): _description_. Defaults to ( "C-H-N-O", "F-Cl-Br-I", "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu", ).
            compute_for_all_elements (bool, optional): _description_. Defaults to True.
            dimensions (Tuple[int], optional): _description_. Defaults to (1, 2).
            min_size (int, optional): _description_. Defaults to 20.
            aggregation_functions (Tuple[str], optional): _description_. Defaults to ("min", "max", "mean", "std").
        """

        atom_types = [] if atom_types is None else atom_types
        self.elements = atom_types
        self.atom_types = (
            list(atom_types) + ["all"] if compute_for_all_elements else list(atom_types)
        )
        self.compute_for_all_elements = compute_for_all_elements
        self.dimensions = dimensions
        self.min_size = min_size
        self.aggregation_functions = aggregation_functions

    def _get_feature_labels(self) -> List[str]:
        labels = []
        for atom_type in self.atom_types:
            for dim in self.dimensions:
                for parameter in ("birth", "death", "persistence"):
                    for aggregation in self.aggregation_functions:
                        labels.append(f"{atom_type}_dim{dim}_{parameter}_{aggregation}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        res = get_diagrams_for_structure(
            structure, self.elements, self.compute_for_all_elements, self.min_size
        )

        flat_results = []
        for atom_type in self.atom_types:
            for dim in self.dimensions:

                dimname = f"dim{dim}"
                stats = persistent_diagram_stats(
                    res[atom_type][dimname], self.aggregation_functions
                )
                flat_stats = list(flatten(stats).values())
                flat_results.extend(flat_stats)

        return np.array(flat_results)

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
