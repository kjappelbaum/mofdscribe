# -*- coding: utf-8 -*-
"""Partial charge histogram featurizer."""
from typing import List

import numpy as np
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.eqeq import get_eqeq_charges
from mofdscribe.featurizers.utils.extend import operates_on_istructure, operates_on_structure
from mofdscribe.featurizers.utils.histogram import get_rdf
from mofdscribe.mof import MOF
from mofdscribe.types import StructureIStructureType

__all__ = ["PartialChargeHistogram"]


@operates_on_istructure
@operates_on_structure
class PartialChargeHistogram(MOFBaseFeaturizer):
    """Compute partial charges using the EqEq charge equilibration method [Ongari2019]_.

    Then derive a fix-length feature vector from the partial charges by binning
    charges in a histogram.
    """

    def __init__(
        self,
        min_charge: float = -4,
        max_charge: float = 4,
        bin_size: float = 0.5,
    ) -> None:
        """Construct a new PartialChargeHistogram featurizer.

        Args:
            min_charge (float): Minimum limit of bin grid.
                Defaults to -4.
            max_charge (float): Maximum limit of bin grid.
                Defaults to 4.
            bin_size (float): Bin size.
                Defaults to 0.5.
        """
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.bin_size = bin_size

    def _get_grid(self):
        return np.arange(self.min_charge, self.max_charge, self.bin_size)

    def feature_labels(self) -> List[str]:
        return [f"chargehist_{val}" for val in self._get_grid()]

    def featurize(self, mof: MOF) -> np.ndarray:
        return self._featurize(s=mof.structure)

    def _featurize(self, s: StructureIStructureType) -> np.ndarray:
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
        return ["Kevin Maik Jablonka", "Daniele Ongari", "Christopher Wilmer"]
