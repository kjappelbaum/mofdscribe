"""Atomic-property weighted autocorrelation function.
See original implementation by authors https://github.com/tomdburns/AP-RDF
"""

from matminer import BaseFeaturizer
import numpy as np
import math
from typing import Tuple, Union
from ..utils.aggregators import AGGREGATORS
from collections import defaultdict
from functools import cached_property


class APRDF(BaseFeaturizer):
    """
    Generalization of descriptor described by Fernandez et al. In the article they describe the product of atomic properties
    \operatorname{RDF}^{p}(R)=f \sum_{i, j}^{\text {all atom puirs }} P_{i} P_{j} \mathrm{e}^{-B\left(r_{ij}-R\right)^{2}}
    Here, we also implement the difference.
    """

    def __init__(
        self,
        cutoff: float = 20.0,
        bin_size: float = 0.1,
        lower_lim: float = 2.0,
        bw: Union[float, None] = 0.1,
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        aggreations: Tuple[str] = ("avg", "product", "diff"),
        property_prod: bool = True,
        property_diff: bool = True,
    ):
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.property_prod = property_prod
        self.property_diff = property_diff
        self.properties = properties
        self.lower_lim = lower_lim
        self.bw = bw
        self.aggregations = aggreations

    def precheck(self):
        pass

    @cached_property
    def _bins(self):
        return np.arange(self.lower_lim, self.cutoff + self.bin_size, self.bin_size)

    def _get_feature_labels(self):
        labels = []
        for property in self.properties:
            for aggregation in self.aggregations:
                for _, bin in enumerate(self._bins):
                    labels.append(f"{property}_{aggregation}_{bin}")

        return labels

    def featurizer(self, s):
        neighbors_lst = s.get_all_neighbors(self.cutoff)

        results = defaultdict(lambda: defaultdict(list))

        # ToDo: This is quite slow. We can, however, only use numba if we do not access the
        # pymatgen object
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

    def feature_labels(self):
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
