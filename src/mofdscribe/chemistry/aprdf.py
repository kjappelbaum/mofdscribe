"""Atomic-property labeled autocorrelation function."""

from matminer import BaseFeaturizer


class APRDF(BaseFeaturizer):
    def __init__(self):
        ...

    def featurizer(self, s):
        ...

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
