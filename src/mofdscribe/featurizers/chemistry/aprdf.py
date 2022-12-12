# -*- coding: utf-8 -*-
"""Atomic-property weighted autocorrelation function.

See alternative implementation https://github.com/tomdburns/AP-RDF (likely
faster as it also has a lower-level implementation)
"""
from functools import cached_property
from typing import List, Tuple, Union

import numpy as np
from element_coder import encode
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer

from ..utils.aggregators import AGGREGATORS
from ..utils.extend import operates_on_istructure, operates_on_structure

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
        bin_size: float = 0.25,
        b_smear: Union[float, None] = 10,
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        aggregations: Tuple[str] = ("avg", "product", "diff"),
        normalize: bool = False,
        primitive: bool = True,
    ):
        """Set up an atomic property (AP) weighted radial distribution function.

        Args:
            cutoff (float): Consider neighbors up to this value (in
                Angstrom). Defaults to 20.0.
            lower_lim (float): Lowest distance (in Angstrom) to consider.
                Defaults to 2.0.
            bin_size (float): Bin size for binning.
                Defaults to 0.25.
            b_smear (Union[float, None]): Band width for Gaussian smearing.
                If None, the unsmeared histogram is used. Defaults to 10.
            properties (Tuple[str, int]): Properties used for calculation of the AP-RDF.
                All properties of `pymatgen.core.Species` are available in
                addition to the integer `1` that will set P_i=P_j=1. Defaults to
                ("X", "electron_affinity").
            aggregations (Tuple[str]): Methods used to combine the
                properties.
                See `mofdscribe.featurizers.utils.aggregators.AGGREGATORS` for available
                options. Defaults to ("avg", "product", "diff").
            normalize (bool): If True, the histogram is normalized by dividing
                by the number of atoms. Defaults to False.
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to True.
        """
        self.lower_lim = lower_lim
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.properties = properties
        self.normalize = normalize

        self.b_smear = b_smear
        self.aggregations = aggregations
        super().__init__(primitive=primitive)

    def precheck(self):
        pass

    @cached_property
    def _bins(self):
        num_bins = int((self.cutoff - self.lower_lim) // self.bin_size)
        bins = np.linspace(self.lower_lim, self.cutoff, num_bins)
        return bins

    def _get_feature_labels(self):
        aprdfs = np.empty(
            (len(self.properties), len(self.aggregations), len(self._bins)), dtype=object
        )
        for pi, prop in enumerate(self.properties):
            for ai, aggregation in enumerate(self.aggregations):
                for bin_index, _ in enumerate(self._bins):
                    aprdfs[pi][ai][bin_index] = f"aprdf_{prop}_{aggregation}_{bin_index}"

        return list(aprdfs.flatten())

    def _featurize(self, s: Union[Structure, IStructure]) -> np.array:
        bins = self._bins
        aprdfs = np.zeros((len(self.properties), len(self.aggregations), len(bins)))

        # todo: use numba to speed up
        for i, item in enumerate(s):
            for j in range(i + 1, len(s)):
                dist = s.get_distance(i, j)
                if dist < self.cutoff and dist > self.lower_lim:
                    bin_idx = int((dist - self.lower_lim) // self.bin_size)
                    for pi, prop in enumerate(self.properties):
                        for ai, agg in enumerate(self.aggregations):
                            p0 = encode(item.specie, prop)
                            p1 = encode(s[j].specie, prop)

                            agg_func = AGGREGATORS[agg]
                            p = agg_func([p0, p1])
                            aprdfs[pi][ai][bin_idx] += p * np.exp(
                                -self.b_smear * (dist - bins[bin_idx]) ** 2
                            )

        if self.normalize:
            aprdfs /= len(s)

        return aprdfs.flatten()

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
