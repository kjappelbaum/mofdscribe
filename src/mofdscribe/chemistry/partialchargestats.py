from ctypes import Structure
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pyeqeq import run_on_cif

from mofdscribe.utils.aggregators import ARRAY_AGGREGATORS


class PartialChargeStats(BaseFeaturizer):
    def __init__(self, aggregtations: Tuple[str] = ("max", "min", "std")) -> None:
        """Compute partial charges using the EqEq charge equilibration method.
        Then derive a fix-length feature vector from the partial charges using aggregative statistics.

        Args:
            aggregtations (Tuple[str], optional): Aggregations to compute. For available methods,
                see :py:met:`mofdscribe.utils.aggregators.ARRAY_AGGREGATORS`. Defaults to ("max", "min", "std").
        """
        self.aggregations = aggregtations

    def feature_labels(self) -> List[str]:
        return [f"charge_{agg}" for agg in self.aggregations]

    def featurize(self, s: Structure) -> np.ndarray:
        with NamedTemporaryFile("w", suffix=".cif") as f:
            s.to("cif", f.name)
            results = run_on_cif(f.name)

        aggregated = [ARRAY_AGGREGATORS[agg](results) for agg in self.aggregations]

        return np.array(aggregated)

    def citations(self) -> List[str]:
        return [
            "@article{Ongari2018,"
            "doi = {10.1021/acs.jctc.8b00669},"
            "url = {https://doi.org/10.1021/acs.jctc.8b00669},"
            "year = {2018},"
            "month = nov,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {15},"
            "number = {1},"
            "pages = {382--401},"
            "author = {Daniele Ongari and Peter G. Boyd and Ozge Kadioglu and Amber K. Mace and Seda Keskin and Berend Smit},"
            "title = {Evaluating Charge Equilibration Methods To Generate Electrostatic Fields in Nanoporous Materials},"
            "journal = {Journal of Chemical Theory and Computation}"
            "}",
            "@article{Wilmer2012,"
            "doi = {10.1021/jz3008485},"
            "url = {https://doi.org/10.1021/jz3008485},"
            "year = {2012},"
            "month = aug,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {3},"
            "number = {17},"
            "pages = {2506--2511},"
            "author = {Christopher E. Wilmer and Ki Chul Kim and Randall Q. Snurr},"
            "title = {An Extended Charge Equilibration Method},"
            "journal = {The Journal of Physical Chemistry Letters}"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
