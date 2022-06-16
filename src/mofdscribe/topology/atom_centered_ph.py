# -*- coding: utf-8 -*-
"""Featurizers using persistent homology -- applied in an atom-centred manner."""
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from moltda.construct_pd import construct_pds
from moltda.vectorize_pds import diagrams_to_arrays
from pymatgen.core import IStructure, Structure

from mofdscribe.utils import flatten
from mofdscribe.utils.aggregators import ARRAY_AGGREGATORS

from ._tda_helpers import persistent_diagram_stats


# Todo: allow doing this with cutoff and coordination shells
class AtomCenteredPHSite(BaseFeaturizer):
    """Site featurizer for atom-centered statistics of persistence diagrams.

    This featurizer is an abstraction of the on described in the work of Jiang
    et al. (2021) [Jiang2021]_. It computes the persistence diagrams for the
    neighborhood of every site and then aggregates the diagrams by computing
    certain statistics.

    To use this featurizer on a complete structure without additional
    resolutions over the chemistry, you can wrap it in a
    :class:`~matminer.featurizers.structure.SiteStatsFingerprint`.

    Examples:
        >>> from matminer.featurizers.structure import SiteStatsFingerprint
        >>> from mofdscribe.topology.atom_centered_ph import AtomCenteredPHSite
        >>> featurizer = SiteStatsFingerprint(AtomCenteredPHSite())
        >>> featurizer.featurize(structure)
        np.array([2,...]) # np.array with features

    However, if you want the additional chemical dimension, you can use the the
    :class:`~mofdscribe.topology.atom_centered_ph.AtomCenteredPH`.
    """

    def __init__(
        self,
        aggregation_functions: Optional[Tuple[str]] = ("min", "max", "mean", "std"),
        cutoff: Optional[float] = 12,
        dimensions: Optional[Tuple[int]] = (1, 2),
    ) -> None:
        """
        Construct a new AtomCenteredPHSite featurizer.

        Args:
            aggregation_functions (Tuple[str], optional): Aggregations to
                compute on the persistence diagrams (over birth/death time and
                persistence). Defaults to ("min", "max", "mean", "std").
            cutoff (float, optional): Consider neighbors of site within this radius (in
                Angstrom). Defaults to 12.
            dimensions (Tuple[int], optional): Betti numbers of consider.
                0 describes isolated components, 1 cycles and 2 cavities.
                Defaults to (1, 2).
        """
        self.aggregation_functions = aggregation_functions
        self.cutoff = cutoff
        self.dimensions = dimensions

    def featurize(self, s: Union[Structure, IStructure], idx: int) -> np.ndarray:
        neighbors = s.get_neighbors(s[idx], self.cutoff)
        neighbor_structure = IStructure.from_sites(neighbors)
        diagrams = construct_pds(neighbor_structure.cart_coords)
        diagrams = diagrams_to_arrays(diagrams)
        results = {}
        for dim in self.dimensions:
            key = f"dim{dim}"
            results[key] = persistent_diagram_stats(diagrams[key], self.aggregation_functions)
        return np.array(list(flatten(results).values()))

    def _get_feature_labels(self) -> List[str]:
        names = []
        for dim in self.dimensions:
            dim_key = f"dim{dim}"
            for parameter in ("birth", "death", "persistence"):
                for aggregation in self.aggregation_functions:
                    names.append(f"{dim_key}_{parameter}_{aggregation}")
        return names

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return [
            "@article{Jiang2021,"
            "doi = {10.1038/s41524-021-00493-w},"
            "url = {https://doi.org/10.1038/s41524-021-00493-w},"
            "year = {2021},"
            "month = feb,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {7},"
            "number = {1},"
            "author = {Yi Jiang and Dong Chen and Xin Chen and Tangyi Li "
            "and Guo-Wei Wei and Feng Pan},"
            "title = {Topological representations of crystalline compounds "
            "for the machine-learning prediction of materials properties},"
            "journal = {npj Computational Materials}"
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


# ToDo: Leverage symmetry to do not recompute for symmetry-equivalent sites
class AtomCenteredPH(BaseFeaturizer):
    """Atom-centered featurizer for persistence diagrams.

    It runs :class:`~mofdscribe.topology.atom_centered_ph.AtomCenteredPH` for every site.

    It aggregates the results over atom types that are specified in the constructor
    via aggregation functions specified in the constructor.
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
        aggregation_functions: Optional[Tuple[str]] = ("min", "max", "mean", "std"),
        species_aggregation_functions: Optional[Tuple[str]] = ("min", "max", "mean", "std"),
        cutoff: Optional[float] = 12,
        dimensions: Optional[Tuple[int]] = (1, 2),
    ) -> None:
        """
        Construct a new AtomCenteredPH featurizer.

        Args:
            atom_types (tuple, optional): Atoms that are used to create substructures
                that are analysed using persistent homology.
                If multiple atom types separated by hash are provided, e.g. "C-H-N-O",
                then the substructure consists of all atoms of type C, H, N, or O.
                Defaults to ( "C-H-N-O", "F-Cl-Br-I",
                "Cu-Mn-Ni-Mo-Fe-Pt-Zn-Ca-Er-Au-Cd-Co-Gd-Na-Sm-Eu-Tb-V-Ag-Nd-U-Ba-Ce-K-Ga-
                "Cr-Al-Li-Sc-Ru-In-Mg-Zr-Dy-W-Yb-Y-Ho-Re-Be-Rb-La-Sn-Cs-Pb-Pr-Bi-Tm-Sr-Ti-Hf-Ir-
                "Nb-Pd-Hg-Th-Np-Lu-Rh-Pu", ).
            aggregation_functions (Tuple[str], optional): Aggregations to compute on the persistence
                diagrams (over birth/death time and persistence).
                Defaults to ("min", "max", "mean", "std").
            species_aggregation_functions (Tuple[str], optional): Aggregations to use to combine
                features derived for sites of a specific atom type, e.g., the site features of all `C-H-N-O`.
                Defaults to ("min", "max", "mean", "std").
            cutoff (float, optional): Consider neighbors of site within this radius (in Angstrom).
                Defaults to 12.
            dimensions (Tuple[int], optional): Betti numbers of consider. 0 describes isolated components,
                1 cycles and 2 cavities. Defaults to (1, 2).
        """
        self.aggregation_functions = aggregation_functions
        self.species_aggregation_functions = species_aggregation_functions
        self.cutoff = cutoff
        self.dimensions = dimensions
        self.site_featurizer = AtomCenteredPHSite(
            aggregation_functions=aggregation_functions, cutoff=cutoff, dimensions=dimensions
        )
        self.atom_types = atom_types

    def _get_relevant_atom_type(self, element: str) -> str:
        for atom_type in self.atom_types:
            if element in atom_type:
                return atom_type

    def featurize(self, s: Union[Structure, IStructure]) -> np.ndarray:
        results = defaultdict(list)
        for idx, site in enumerate(s):
            atom_type = self._get_relevant_atom_type(site.specie.symbol)
            features = self.site_featurizer.featurize(s, idx)
            results[atom_type].append(features)

        long_results = []
        for atom_type in self.atom_types:
            if atom_type not in results:
                long_results.extend(
                    np.zeros(
                        len(self.site_featurizer.feature_labels())
                        * len(self.species_aggregation_functions)
                    )
                )
            else:
                v = np.array(results[atom_type])
                for aggregation in self.species_aggregation_functions:
                    agg_func = ARRAY_AGGREGATORS[aggregation]
                    long_results.extend(agg_func(v, axis=0))

        return np.array(long_results)

    def _get_feature_labels(self) -> List[str]:
        names = []
        for atom_type in self.atom_types:
            for aggregation in self.species_aggregation_functions:
                for fl in self.site_featurizer.feature_labels():
                    names.append(f"{atom_type}_{aggregation}_{fl}")
        return names

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return [
            "@article{Jiang2021,"
            "doi = {10.1038/s41524-021-00493-w},"
            "url = {https://doi.org/10.1038/s41524-021-00493-w},"
            "year = {2021},"
            "month = feb,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {7},"
            "number = {1},"
            "author = {Yi Jiang and Dong Chen and Xin Chen and Tangyi Li and Guo-Wei Wei and Feng Pan},"
            "title = {Topological representations of crystalline compounds for "
            "the machine-learning prediction of materials properties},"
            "journal = {npj Computational Materials}"
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
