# -*- coding: utf-8 -*-
"""Guest-centered atomic-property weighted autocorrelation function."""
from functools import cached_property
from typing import List, Optional, Tuple, Union

import numpy as np
from element_coder import encode
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.hostguest.utils import HostGuest, _extract_host_guest

from ..utils.aggregators import AGGREGATORS

__all__ = ["GuestCenteredAPRDF"]


class GuestCenteredAPRDF(BaseFeaturizer):
    """Guest-centered atomic-property weighted autocorrelation function.

    This is a modification of the AP-RDF that is centered on the guest atoms.
    That is, we only consider the distances between the guest atoms and the host atoms (within the cutoffs).

    For more details about the AP-RDF, see `mofdscribe.featurizers.chemistry.aprdf.APRDF`.
    """

    def __init__(
        self,
        cutoff: float = 20.0,
        lower_lim: float = 2,
        bin_size: float = 0.25,
        b_smear: Union[float, None] = 10,
        properties: Tuple[str, int] = ("X", "electron_affinity"),
        aggregations: Tuple[str] = ("avg", "product", "diff"),
        local_env_method: str = "vesta",
        normalize: bool = False,
    ):
        """Set up an atomic property (AP) weighted radial distribution function.

        Args:
            cutoff (float): Consider neighbors up to this value (in
                Angstrom). Defaults to 20.0.
            lower_lim (float): Lowest distance (in Angstrom) to consider.
                Defaults to 2.
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
            local_env_method (str): Method used to compute the structure graph.
            normalize (bool): If True, the histogram is normalized by dividing
                by the number of atoms. Defaults to False.
        """
        self.lower_lim = lower_lim
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.properties = properties
        self._local_env_method = local_env_method
        self.b_smear = b_smear
        self.aggregations = aggregations
        self.normalize = normalize

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
                    aprdfs[pi][ai][bin_index] = f"guest_aprdf_{prop}_{aggregation}_{bin_index}"

        return list(aprdfs.flatten())

    def _extract_host_guest(
        self,
        structure: Optional[Union[Structure, IStructure]] = None,
        host_guest: Optional[HostGuest] = None,
    ):
        return _extract_host_guest(
            structure=structure,
            host_guest=host_guest,
            remove_guests=True,
            operates_on="molecule",
            local_env_method=self._local_env_method,
        )

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

        bins = self._bins
        aprdfs = np.zeros((len(self.properties), len(self.aggregations), len(bins)))

        # todo: use numba to speed up
        for _i, guest in enumerate(flattened_guests):
            guest_frac_coords = host_guest.host.lattice.get_fractional_coords(guest.coords)
            for _j, site in enumerate(host_guest.host):
                dist, _ = host_guest.host.lattice.get_distance_and_image(
                    guest_frac_coords, site.frac_coords
                )
                if dist < self.cutoff and dist > self.lower_lim:
                    bin_idx = int((dist - self.lower_lim) // self.bin_size)
                    for pi, prop in enumerate(self.properties):
                        for ai, agg in enumerate(self.aggregations):
                            p0 = encode(guest.specie, prop)
                            p1 = encode(site.specie, prop)

                            agg_func = AGGREGATORS[agg]
                            p = agg_func([p0, p1])
                            aprdfs[pi][ai][bin_idx] += p * np.exp(
                                -self.b_smear * (dist - bins[bin_idx]) ** 2
                            )

        if self.normalize:
            aprdfs /= len(host_guest.guests)

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
