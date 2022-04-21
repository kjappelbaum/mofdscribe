# -*- coding: utf-8 -*-
"""Implements persistent homology images"""
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from ._tda_helpers import get_min_max_from_dia


class PHImage(BaseFeaturizer):
    """
    `Adams et al. (2017) <https://www.jmlr.org/papers/volume18/16-337/16-337.pdf>`_
    introduced a stable vector representation of persistent homology.

    In persistent images, one replaces birth–persistence pairs (b, d – b)
    by a Gaussians (to spread its influence across the neighborhood,
    since nearby points represent features of similar size).
    Additionally, one multiplies with special weighting function such that

    .. math::

        f(x, y)=w(y) \sum_{(b, d) \in D_{g}(P)} N((b, d-b), \sigma)

    A common weighting function is the linear function :math:`w(y) = y`.

    One application for porous materials has been reported by
    `Aditi S. Krishnapriyan et al. (2017) <https://pubs.acs.org/doi/full/10.1021/acs.jpcc.0c01167>`_.

    Typically, persistent images are computed for all atoms in the structure.
    However, one can also compute persistent images for a subset of atoms.
    This can be done by specifying the atom types in the constructor.
    """

    def __init__(
        self,
        atom_types: Tuple[str] = (
            "C-H-N-O",
            "F-Cl-Br-I",
            "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu",
        ),
        dimensions: Tuple[int] = (0, 1, 2),
        compute_for_all_elements: bool = True,
        min_size: int = 10,
        image_size: Tuple[int] = (20, 20),
        spread: float = 0.2,
        weight: str = "identity",
        max_B: Union[int, List[int]] = 18,
        max_P: Union[int, List[int]] = 18,
        max_fit_tolerence: float = 0.1,
    ) -> None:

        self.atom_types = atom_types
        self.compute_for_all_elements = compute_for_all_elements
        self.min_size = min_size
        self.image_size = image_size
        self.spread = spread
        self.dimensions = dimensions
        self.weight = weight
        if isinstance(max_B, (list, tuple)):
            assert len(max_B) == len(
                dimensions
            ), "max_B must be a list of length equal to the number of dimensions"
        else:
            max_B = [max_B] * len(dimensions)

        if isinstance(max_P, (list, tuple)):
            assert len(max_P) == len(
                dimensions
            ), "max_P must be a list of length equal to the number of dimensions"
        else:
            max_P = [max_P] * len(dimensions)

        max_P_ = [0, 0, 0, 0]
        max_B_ = [0, 0, 0, 0]
        for dim in dimensions:
            max_P_[dim] = max_P[dim]
            max_B_[dim] = max_B[dim]

        self.max_B = max_B_
        self.max_P = max_P_

        self.max_fit_tolerance = max_fit_tolerence

    def _get_feature_labels(self) -> List[str]:
        labels = []
        _elements = list(self.atom_types)
        if self.compute_for_all_elements:
            _elements.append("all")
        for element in _elements:
            for dim in self.dimensions:
                for pixel_a in range(self.image_size[0]):
                    for pixel_b in range(self.image_size[1]):
                        labels.append(f"pi_{element}_{dim}_{pixel_a}_{pixel_b}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        results = get_persistent_images_for_structure(
            structure,
            elements=self.atom_types,
            compute_for_all_elements=self.compute_for_all_elements,
            min_size=self.min_size,
            pixels=self.image_size,
            spread=self.spread,
            weighting=self.weight,
            maxB=self.max_B,
            maxP=self.max_P,
        )
        features = []
        elements = list(self.atom_types)
        if self.compute_for_all_elements:
            elements.append("all")
        for element in elements:
            for dim in self.dimensions:

                features.append(np.array(results["image"][element][dim]).flatten())
        return np.concatenate(features)

    def fit(self, structures: List[Union[Structure, IStructure]]) -> None:
        """
        Find the limits (maximum/minimum birth/death and persistence)
        for all the structures in the dataset and store them in the object.

        Args:
            structures (List[Union[Structure, IStructure]]): List of structures to find the limits for.
        """
        if not isinstance(structures, (list, tuple)):
            structures = [structures]

        limits = defaultdict(list)

        for structure in structures:
            lim = get_persistence_image_limits_for_structure(
                structure, self.atom_types, self.compute_for_all_elements, self.min_size
            )
            for k, v in lim.items():
                limits[k].extend(v)

        # birth min, max persistence min, max
        maxP = []
        maxB = []

        for k, v in limits.items():
            v = np.array(v)
            mB = np.max(v[:, 1])
            mP = np.max(v[:, 3])
            maxB.append(mB + self.max_fit_tolerance * mB)
            maxP.append(mP + self.max_fit_tolerance * mP)

        self.max_B = maxB
        self.max_P = maxP

    def citations(self) -> List[str]:
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
            "@article{krishnapriyan_machine_2021,"
            "title={Machine learning with persistent homology and chemical word embeddings improves prediction accuracy and interpretability in metal-organic frameworks},"
            "author={Krishnapriyan, Aditi S and Montoya, Joseph and Haranczyk, Maciej and Hummelsh{\o}j, Jens and Morozov, Dmitriy},"
            "journal = {Scientific Reports},"
            "volume = {11},"
            "numer = {1},"
            "issn = {2045-2322},"
            "pages = {8888},"
            "year={2021},"
            "doi = {10.1038/s41598-021-88027-8}"
            "}",
            "@article{adams2017persistence,"
            "title={Persistence images: A stable vector representation of persistent homology},"
            "author={Adams, Henry and Emerson, Tegan and Kirby, Michael and Neville, Rachel and Peterson, Chris and Shipman, Patrick and Chepushtanova, Sofya and Hanson, Eric and Motta, Francis and Ziegelmeier, Lori},"
            "journal={Journal of Machine Learning Research},"
            "volume={18},"
            "year={2017}"
            "}",
        ]

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]
