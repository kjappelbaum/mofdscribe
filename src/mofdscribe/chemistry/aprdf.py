"""Atomic-property weighted autocorrelation function.
See original implementation by authors https://github.com/tomdburns/AP-RDF
"""

from matminer import BaseFeaturizer
import numpy as np


class APRDF(BaseFeaturizer):
    """
    \operatorname{RDF}^{p}(R)=f \sum_{i, j}^{\text {all atom puirs }} P_{i} P_{j} \mathrm{e}^{-B\left(\tau_{j}-R\right)^{2}}
    """

    def __init__(self, cutoff: float = 20.0, bin_size: float = 0.1):
        self.cutoff = cutoff
        self.bin_size = bin_size

    def featurizer(self, s):
        neighbors_lst = s.get_all_neighbors(self.cutoff)

    def feature_labels(self):
        ...

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
