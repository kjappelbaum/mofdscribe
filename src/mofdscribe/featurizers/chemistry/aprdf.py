# -*- coding: utf-8 -*-
"""Atomic-property weighted autocorrelation function.

See alternative implementation https://github.com/tomdburns/AP-RDF (likely
faster as it also has a lower-level implementation)
"""

from collections import defaultdict
from functools import cached_property
from typing import List, Tuple, Union

import numpy as np
from element_coder import encode
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer

from ..utils.aggregators import AGGREGATORS
from ..utils.extend import operates_on_istructure, operates_on_structure
from ..utils.histogram import get_rdf, smear_histogram

__all__ = ["APRDF"]


@operates_on_structure
@operates_on_istructure
class APRDF(MOFBaseFeaturizer):
    r"""Generalization of descriptor described by `Fernandez et al. <https://pubs.acs.org/doi/10.1021/jp404287t>`_.

    In the article they describe the product of atomic properties as weighting
    of a "conventional" radiual distribution function "RDF".

    .. math::
        \operatorname{RDF}^{p}(R)=f \sum_{i, j}^{\text {all atom puirs }} P_{i} P_{j} \mathrm{e}^{-B\left(r_{ij}-R\right)^{2}} # noqa: E501

    Here, we allow for a wider choice of option for aggregation of properties
    :math:`P_{i}` and :math:`P_{j}` (not only the product).

    Examples:
        >>> from mofdscribe.chemistry.aprdf import APRDF
        >>> from pymatgen.core.structure import Structure
        >>> s = Structure.from_file("tests/test_files/LiTiO3.cif")
        >>> aprdf = APRDF()
        >>> features = aprdf.featurize(s)
    """

    def __init__(
        self,
        cutoff: float = 20.0,
        lower_lim: float = 2.0,
        bin_size: float = 0.1,
        bw: Union[float, None] = 0.1,
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        aggregations: Tuple[str] = ("avg", "product", "diff"),
        primitive: bool = True,
    ):
        """Set up an atomic property (AP) weighted radial distribution function.

        Args:
            cutoff (float): Consider neighbors up to this value (in
                Angstrom). Defaults to 20.0.
            lower_lim (float): Lowest distance (in Angstrom) to consider.
                Defaults to 2.0.
            bin_size (float): Bin size for binning.
                Defaults to 0.1.
            bw (Union[float, None]): Band width for Gaussian smearing.
                If None, the unsmeared histogram is used. Defaults to 0.1.
            properties (Tuple[str, int]): Properties used for calculation of the AP-RDF.
                All properties of `pymatgen.core.Species` are available in
                addition to the integer `1` that will set P_i=P_j=1. Defaults to
                ("X", "electron_affinity").
            aggregations (Tuple[str]): Methods used to combine the
                properties.
                See `mofdscribe.featurizers.utils.aggregators.AGGREGATORS` for available
                options. Defaults to ("avg", "product", "diff").
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to True.
        """
        self.lower_lim = lower_lim
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.properties = properties

        self.bw = bw
        self.aggregations = aggregations
        super().__init__(primitive=primitive)

    def precheck(self):
        pass

    @cached_property
    def _bins(self):
        return np.arange(self.lower_lim, self.cutoff, self.bin_size)

    def _get_feature_labels(self):
        labels = []
        for prop in self.properties:
            for aggregation in self.aggregations:
                for _, bin_ in enumerate(self._bins):
                    labels.append(f"aprdf_{prop}_{aggregation}_{bin_}")

        return labels

    def _featurize(self, s: Union[Structure, IStructure]) -> np.array:
        neighbors_lst = s.get_all_neighbors(self.cutoff)

        results = defaultdict(lambda: defaultdict(list))

        # ToDo: This is quite slow. We can, however, only use numba if we do not access the
        # pymatgen object
        # for numba, we could make one "slow" loop where we store everything we need
        # in one/two arrays and then we make the N*N loop
        for i, site in enumerate(s):
            site_neighbors = neighbors_lst[i]
            for n in site_neighbors:
                if n.nn_distance > self.lower_lim:
                    for prop in self.properties:
                        if prop in ("I", 1):
                            p0 = 1
                            p1 = 1
                        else:
                            p0 = encode(site.specie.symbol, prop)
                            p1 = encode(n.specie.symbol, prop)
                        for agg in self.aggregations:
                            agg_func = AGGREGATORS[agg]
                            results[prop][agg].append(agg_func((p0, p1)) * n.nn_distance)

        feature_vec = []
        for prop in self.properties:
            for aggregation in self.aggregations:
                rdf = get_rdf(
                    results[prop][aggregation],
                    self.lower_lim,
                    self.cutoff,
                    self.bin_size,
                    s.num_sites,
                    s.volume,
                )
                if self.bw is not None:
                    rdf = smear_histogram(rdf, self.bw, self.lower_lim, self.cutoff + self.bin_size)

                feature_vec.append(rdf)

        return np.concatenate(feature_vec)

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def citations(self) -> List[str]:
        return [
            "@article{Fernandez2013,"
            "doi = {10.1021/jp404287t},"
            "url = {https://doi.org/10.1021/jp404287t},"
            "year = {2013},"
            "month = jul,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {117},"
            "number = {27},"
            "pages = {14095--14105},"
            "author = {Michael Fernandez and Nicholas R. Trefiak and Tom K. Woo},"
            "title = {Atomic Property Weighted Radial Distribution Functions "
            "Descriptors of Metal{\textendash}Organic Frameworks for the Prediction "
            "of Gas Uptake Capacity},"
            "journal = {The Journal of Physical Chemistry C}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
