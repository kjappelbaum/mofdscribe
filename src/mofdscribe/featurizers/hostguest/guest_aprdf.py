# -*- coding: utf-8 -*-
"""Guest-centered atomic-property weighted autocorrelation function.
"""
from typing import Union, Tuple, List, Optional
from mofdscribe.featurizers.base import MOFBaseFeaturizer
from ..utils.extend import operates_on_istructure, operates_on_structure

from mofdscribe.featurizers.chemistry.aprdf import compute_aprdf_dict, flatten_aprdf
import numpy as np
from functools import cached_property
from pymatgen.core import IStructure, Structure, Site

from mofdscribe.featurizers.hostguest.utils import HostGuest, _extract_host_guest
from matminer.featurizers.base import BaseFeaturizer


class GuestCenteredAPRDF(BaseFeaturizer):
    def __init__(self,
        cutoff: float = 20.0,
        lower_lim: float = 0,
        bin_size: float = 1,
        bw: Union[float, None] = 0.1,
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        aggregations: Tuple[str] = ("avg", "product", "diff"),
        local_env_method: str = "vesta",
    ):
        """Set up an atomic property (AP) weighted radial distribution function.

        Args:
            cutoff (float): Consider neighbors up to this value (in
                Angstrom). Defaults to 20.0.
            lower_lim (float): Lowest distance (in Angstrom) to consider.
                Defaults to 0.5.
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
            local_env_method (str): Method used to compute the structure graph.
        """
        self.lower_lim = lower_lim
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.properties = properties
        self._local_env_method = local_env_method
        self.bw = bw
        self.aggregations = aggregations
  
    @cached_property
    def _bins(self):
        return np.arange(self.lower_lim, self.cutoff, self.bin_size)

    def _get_feature_labels(self):
        labels = []
        for prop in self.properties:
            for aggregation in self.aggregations:
                for _, bin_ in enumerate(self._bins):
                    labels.append(f"host_guest_aprdf_{prop}_{aggregation}_{bin_}")

        return labels


    def _extract_host_guest(
        self,
        structure: Optional[Union[Structure, IStructure]] = None,
        host_guest: Optional[HostGuest] = None,
    ):
       return _extract_host_guest(structure=structure, host_guest=host_guest, remove_guests=True, operates_on='molecule', local_env_method=self._local_env_method)

    def featurize(
        self,
        structure: Optional[Union[Structure, IStructure]],
        host_guest: Optional[HostGuest] = None,
    ) -> np.ndarray:
        """
        Compute the features of the host and the guests and aggregate them.

        Args:
            structure (Optional[Union[Structure, IStructure]]): The structure to featurize.
            host_guest (Optional[HostGuest]): The host_guest to featurize.
                If you provide this, you must not provide structure.

        Returns:
            np.ndarray: The features of the host and the guests.

        Raises:
            ValueError: If we cannot detect a host.
        """
        host_guest = self._extract_host_guest(structure=structure, host_guest=host_guest)

        if not host_guest.host:
            raise ValueError(
                "Did not find a framework. This is required for the host guest featurizer."
            )

        if not host_guest.guests:
            raise ValueError(
                "Did not find any guests. This is required for the host guest featurizer."
            )

        flattened_guests = []
        for guest in host_guest.guests:
            flattened_guests.extend(guest.sites)

        neighbors= [host_guest.host.get_sites_in_sphere(site.coords, self.cutoff) for site in flattened_guests]

        results = compute_aprdf_dict(flattened_guests, neighbors, self.properties, self.aggregations, self.lower_lim)
     
        return flatten_aprdf(results, self.properties, self.aggregations, self.lower_lim, self.cutoff, self.bin_size, self.bw, host_guest.host)



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
