"""Atomic-property weighted autocorrelation function.
See alternative implementation https://github.com/tomdburns/AP-RDF (likely faster as it also has a lower-level implementation)
"""

from matminer import BaseFeaturizer
import numpy as np
from typing import Tuple, Union, List
from ..utils.aggregators import AGGREGATORS
from collections import defaultdict
from functools import cached_property
from ..utils.histogram import get_rdf, smear_histogram


class APRDF(BaseFeaturizer):
    """
    Generalization of descriptor described by Fernandez et al. In the article they describe the product of atomic properties
    \operatorname{RDF}^{p}(R)=f \sum_{i, j}^{\text {all atom puirs }} P_{i} P_{j} \mathrm{e}^{-B\left(r_{ij}-R\right)^{2}}
    Here, we also implement the difference.
    """

    def __init__(
        self,
        cutoff: float = 20.0,
        lower_lim: float = 2.0,
        bin_size: float = 0.1,
        bw: Union[float, None] = 0.1,
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        aggreations: Tuple[str] = ("avg", "product", "diff"),
    ):
        """Set up an atomic property (AP) weighted radial distribution function.

        Args:
            cutoff (float, optional): Consider neighbors up to this value (in Angstrom). Defaults to 20.0.
            lower_lim (float, optional): Lowest distance (in Angstrom) to consider. Defaults to 2.0.
            bin_size (float, optional): Bin size for binning. Defaults to 0.1.
            bw (Union[float, None], optional): Band width for Gaussian smearing.
                If None, the unsmeared histogram is used. Defaults to 0.1.
            properties (Tuple[str, int], optional): Properties used for calculation of the AP-RDF.
                All properties of `pymatgen.core.Species` are available
                in addition to the integer `1` that will set P_i=P_j=1.
                Defaults to ("X", "electron_affinity").
            aggreations (Tuple[str], optional): Methods used to combine the properties.
                See `mofdscribe.utils.aggregators.AGGREGATORS` for available options.
                Defaults to ("avg", "product", "diff").
        """
        self.lower_lim = lower_lim
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.properties = properties

        self.bw = bw
        self.aggregations = aggreations

    def precheck(self):
        pass

    @cached_property
    def _bins(self):
        return np.arange(self.lower_lim, self.cutoff + self.bin_size, self.bin_size)

    def _get_feature_labels(self):
        labels = []
        for prop in self.properties:
            for aggregation in self.aggregations:
                for _, bin in enumerate(self._bins):
                    labels.append(f"{prop}_{aggregation}_{bin}")

        return labels

    def featurizer(self, s) -> np.array:
        neighbors_lst = s.get_all_neighbors(self.cutoff)

        results = defaultdict(lambda: defaultdict(list))

        # ToDo: This is quite slow. We can, however, only use numba if we do not access the
        # pymatgen object
        # for numba, we could make one "slow" loop where we store everything we need in one/two arrays and then we make the N*N loop
        for i, site in enumerate(s):
            site_neighbors = neighbors_lst[i]
            for n in site_neighbors:
                if n.nn_distance > self.lower_lim:
                    print(n.nn_distance)
                    for prop in self.properties:
                        if prop == 1:
                            p0 = 1
                            p1 = 1
                        else:
                            p0 = getattr(site.specie, prop)
                            p1 = getattr(n.specie, prop)
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

        return feature_vec.flatten()

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def citations(self):
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
            "title = {Atomic Property Weighted Radial Distribution Functions Descriptors of Metal{\textendash}Organic Frameworks for the Prediction of Gas Uptake Capacity},"
            "journal = {The Journal of Physical Chemistry C}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
