# -*- coding: utf-8 -*-
"""Featurizers using persistent homology -- vectorized using Gaussian mixture models."""
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from pervect import PersistenceVectorizer
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)

from ._tda_helpers import get_diagrams_for_structure


def _apply_and_fill(transformer_func, diagrams):
    applicable_diagrams = []
    rows_to_be_filled = []
    ok_rows = []
    for i, diagram in enumerate(diagrams):
        if diagram.shape[0] > 0:
            applicable_diagrams.append(diagram)
            ok_rows.append(i)
        else:
            rows_to_be_filled.append(i)
    results = transformer_func(applicable_diagrams)
    complete_results = np.zeros((len(diagrams), results.shape[1]))
    complete_results[ok_rows, :] = results
    if len(complete_results) != len(diagrams):
        raise ValueError(
            "Unexpected error. Number of feature vectors is not equal to number of structures"
        )
    return complete_results


def _fit_transform_structures(
    transformers,
    structures,
    atom_types: Tuple[str],
    compute_for_all_elements: bool,
    min_size: int,
    periodic: bool = False,
    no_supercell: bool = False,
    alpha_weight: Optional[str] = None,
):
    logger.info(f"Computing diagrams for {len(structures)} structures")
    diagrams = defaultdict(lambda: defaultdict(list))
    for structure in structures:
        res = get_diagrams_for_structure(
            structure,
            atom_types,
            compute_for_all_elements,
            min_size,
            periodic=periodic,
            no_supercell=no_supercell,
            alpha_weighting=alpha_weight,
        )
        for element, element_d in res.items():
            for dim, dim_d in element_d.items():
                diagrams[element][dim].append(dim_d)

    results = defaultdict(lambda: defaultdict(list))
    for element, element_transformers in transformers.items():
        for dim, transformer in element_transformers.items():
            if len(diagrams[element][dim]) == 0:
                raise ValueError(f"{element} dimension {dim} has no diagrams")
            try:

                results[element][dim] = _apply_and_fill(
                    transformer.fit_transform, diagrams[element][dim]
                )

                if not len(results[element][dim]) == len(structures):
                    raise ValueError(
                        "Unexpected error. Number of feature vectors is not equal to number of structures"
                    )

            except Exception as e:
                logger.error(
                    f"Error fitting transformer: {element} {dim} {diagrams[element][dim]}", e
                )

    return transformers, results


def _transform_structures(
    transformers,
    structures,
    atom_types: Tuple[str],
    compute_for_all_elements: bool,
    min_size: int,
    periodic: bool = False,
    no_supercell: bool = False,
    alpha_weight: Optional[str] = None,
):
    diagrams = defaultdict(lambda: defaultdict(list))
    for structure in structures:
        res = get_diagrams_for_structure(
            structure,
            atom_types,
            compute_for_all_elements,
            min_size,
            periodic=periodic,
            no_supercell=no_supercell,
            alpha_weighting=alpha_weight,
        )
        for element, element_d in res.items():
            for dim, dim_d in element_d.items():
                diagrams[element][dim].append(dim_d)

    results = defaultdict(lambda: defaultdict(list))

    for element, element_transformers in transformers.items():
        for dim, transformer in element_transformers.items():

            results[element][dim] = _apply_and_fill(
                transformer.fit_transform, diagrams[element][dim]
            )

    return results


@operates_on_imolecule
@operates_on_molecule
@operates_on_istructure
@operates_on_structure
class PHVect(MOFBaseFeaturizer):
    """Vectorizer for Persistence Diagrams (PDs) using Gaussian mixture models.

    The vectorization of a diagram is then the weighted maximum likelihood
    estimate of the mixture weights for the learned components given the diagram.

    Importantly, the vectorizations can still be used to compute approximate
    Wasserstein distances.
    """

    def __init__(
        self,
        atom_types: Tuple[str] = (
            "C-H-N-O",
            "F-Cl-Br-I",
            "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V"
            "-Ag-Nd-U-Ba-Ce-K-Ga-Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-"
            "Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-Nb-Pd-Hg-"
            "Th-Np-Lu-Rh-Pu",
        ),
        compute_for_all_elements: bool = True,
        dimensions: Tuple[int] = (1, 2),
        min_size: int = 20,
        n_components: int = 20,
        apply_umap: bool = False,
        umap_n_components: int = 2,
        umap_metric: str = "hellinger",
        p: int = 1,
        random_state: Optional[int] = None,
        periodic: bool = False,
        no_supercell: bool = False,
        primitive: bool = False,
        alpha_weight: Optional[str] = None,
    ) -> None:
        """Construct a PHVect instance.

        Args:
            atom_types (tuple): Atoms that are used to create substructures
                that are analysed using persistent homology.
                If multiple atom types separated by hash are provided,
                e.g. "C-H-N-O", then the substructure consists of all atoms of type C, H, N, or O.
                Defaults to ( "C-H-N-O", "F-Cl-Br-I",
                "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-
                Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-
                Hf-Ir-Nb-Pd-Hg-Th-Np-Lu-Rh-Pu", ).
            compute_for_all_elements (bool): If true, compute persistence images
                for full structure (i.e. with all elements). If false, it will only do it
                for the substructures specified with `atom_types`. Defaults to True.
            dimensions (Tuple[int]): Dimensions of topological features to consider
                for persistence images. Defaults to (1, 2).
            min_size (int): Minimum supercell size (in Angstrom).
                Defaults to 20.
            n_components (int): The number of components or dimensions to use
                in the vectorized representation. Defaults to 20.
            apply_umap (bool):  Whether to apply UMAP to the results to generate
                a low dimensional Euclidean space representation of the diagrams.
                Defaults to False.
            umap_n_components (int):  The number of dimensions of euclidean space
                to use when representing the diagrams via UMAP. Defaults to 2.
            umap_metric (str):  What metric to use for the UMAP embedding if ``apply_umap`` is enabled
                (this option will be ignored if ``apply_umap`` is ``False``).
                Should be one of:
                * ``"wasserstein"``
                * ``"hellinger"``
                Note that if ``"wasserstein"`` is used then transforming new data will not be possible.
                Defaults to "hellinger".
            p (int):  The p in the p-Wasserstein distance to compute.
            random_state (_type_): random state propagated to the Gaussian mixture models (and UMAP).
                Defaults to None.
            periodic (bool): If true, then periodic Euclidean is used in the analysis (experimental!).
                Defaults to False.
            no_supercell (bool): If true, then the supercell is not created.
                Defaults to False.
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to False.
            alpha_weight (Optional[str]): If specified, the use weighted alpha shapes,
                i.e., replacing the points with balls of varying radii.
                For instance `atomic_radius_calculated` or `van_der_waals_radius`.
        """
        atom_types = [] if atom_types is None else atom_types
        self.elements = atom_types
        self.atom_types = (
            list(atom_types) + ["all"] if compute_for_all_elements else list(atom_types)
        )
        self.compute_for_all_elements = compute_for_all_elements
        self.min_size = min_size
        self.dimensions = dimensions
        self.transformers = defaultdict(lambda: defaultdict(dict))
        for atom_type in self.atom_types:
            for dim in self.dimensions:
                self.transformers[atom_type][f"dim{dim}"] = PersistenceVectorizer(
                    n_components=n_components,
                    apply_umap=apply_umap,
                    umap_n_components=umap_n_components,
                    umap_metric=umap_metric,
                    p=p,
                    random_state=random_state,
                )
        self.n_components = n_components
        self.apply_umap = apply_umap
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.random_state = random_state
        self._fitted = False
        self.periodic = periodic
        self.no_supercell = no_supercell
        self.alpha_weight = alpha_weight
        super().__init__(primitive=primitive)

    def _get_feature_labels(self) -> List[str]:
        labels = []
        for atom_type in self.atom_types:
            for dim in self.dimensions:
                for i in range(self.n_components):
                    labels.append(f"phvect_{atom_type}_{dim}_{i}")

        return labels

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def _featurize(
        self, structure: Union[Structure, IStructure, Molecule, IMolecule]
    ) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Must call fit before featurizing")
        res = _transform_structures(
            self.transformers,
            [structure],
            self.elements,
            self.compute_for_all_elements,
            self.min_size,
            periodic=self.periodic,
            no_supercell=self.no_supercell,
            alpha_weight=self.alpha_weight,
        )
        compiled_results = self._reshape_results(res, 1).flatten()
        return compiled_results

    def fit(self, structures: Union[Structure, IStructure, Molecule, IMolecule]) -> "PHVect":
        if self.primitive:
            structures = self._get_primitive_many(structures)
        self.transformers, _ = _fit_transform_structures(
            self.transformers,
            structures,
            self.elements,
            self.compute_for_all_elements,
            self.min_size,
            periodic=self.periodic,
            no_supercell=self.no_supercell,
            alpha_weight=self.alpha_weight,
        )
        self._fitted = True
        return self

    def _reshape_results(self, results, num_structures) -> np.ndarray:
        compiled_results = np.zeros((num_structures, len(self._get_feature_labels())))
        n_col = 0
        for _, element_results in results.items():
            for _, result in element_results.items():
                compiled_results[:, n_col : n_col + self.n_components] = result
                n_col += self.n_components
        return compiled_results

    def fit_transform(
        self, structures: Union[Structure, IStructure, Molecule, IMolecule]
    ) -> np.ndarray:
        if self.primitive:
            structures = self._get_primitive_many(structures)
        self.transformers, results = _fit_transform_structures(
            self.transformers,
            structures,
            self.elements,
            self.compute_for_all_elements,
            self.min_size,
            periodic=self.periodic,
            no_supercell=self.no_supercell,
            alpha_weight=self.alpha_weight,
        )
        compiled_results = self._reshape_results(results, len(structures))
        self._fitted = True
        return compiled_results

    def citations(self):
        return [
            "@article{perea2019approximating,"
            "title   = {Approximating Continuous Functions on Persistence "
            "Diagrams Using Template Functions},"
            "author  = {Jose A. Perea and Elizabeth Munch and Firas A. Khasawneh},"
            "year    = {2019},"
            "journal = {arXiv preprint arXiv: Arxiv-1902.07190}"
            "}",
            "@article{tymochko2019adaptive,"
            "title   = {Adaptive Partitioning for Template Functions on Persistence Diagrams},"
            "author  = {Sarah Tymochko and Elizabeth Munch and Firas A. Khasawneh},"
            "year    = {2019},"
            "journal = {arXiv preprint arXiv: Arxiv-1910.08506}"
            "}",
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
        return ["Kevin Maik Jablonka", "Aditi Krishnapriyan"]
