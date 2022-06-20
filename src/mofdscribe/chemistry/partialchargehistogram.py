# -*- coding: utf-8 -*-
"""Partial charge histogram featurizer."""
from typing import List, Optional, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.eqeq import get_eqeq_charges
from mofdscribe.utils.histogram import get_rdf

__all__ = ["PartialChargeHistogram"]


class PartialChargeHistogram(BaseFeaturizer):
    """Compute partial charges using the EqEq charge equilibration method.

    Then derive a fix-length feature vector from the partial charges by binning
    charges in a histogram.
    """

    def __init__(
        self,
        min_charge: Optional[float] = -4,
        max_charge: Optional[float] = 4,
        bin_size: Optional[float] = 0.5,
    ) -> None:
        """Construct a new PartialChargeHistogram featurizer.

        Args:
            min_charge (float, optional): Minimum limit of bin grid.
                Defaults to -4.
            max_charge (float, optional): Maximum limit of bin grid.
                Defaults to 4.
            bin_size (float, optional): Bin size.
                Defaults to 0.5.
        """
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.bin_size = bin_size

    def _get_grid(self):
        return np.arange(self.min_charge, self.max_charge, self.bin_size)

    def feature_labels(self) -> List[str]:
        return [f"chargehist_{val}" for val in self._get_grid()]

    def featurize(self, s: Union[Structure, IStructure]) -> np.ndarray:
        if isinstance(s, Structure):
            s = IStructure.from_sites(s.sites)
        _, results = get_eqeq_charges(s)

        hist = get_rdf(results, self.min_charge, self.max_charge, self.bin_size, None, None, False)
        return hist

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
            "author = {Daniele Ongari and Peter G. Boyd and Ozge Kadioglu and "
            "Amber K. Mace and Seda Keskin and Berend Smit},"
            "title = {Evaluating Charge Equilibration Methods To Generate "
            "Electrostatic Fields in Nanoporous Materials},"
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
