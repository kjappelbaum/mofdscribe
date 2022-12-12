# -*- coding: utf-8 -*-
"""Compute features on the host and the guests and then aggregate them."""
from typing import Collection, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.hostguest.utils import HostGuest, _extract_host_guest
from mofdscribe.featurizers.utils import set_operates_on
from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS


# ToDo: How do we handle if there is no guest?
# make it optional to remove guests from the structure?
class HostGuestFeaturizer(BaseFeaturizer):
    """
    Compoute features on the host and the guests and then aggregate them.

    This can be useful if you use structures with adsorbed guest molecules as input and want
    to compute adsorption properties such as adsorption energies.

    .. warning::
        Note that we assume that the guest is not covalently bonded to the host.

    .. note::
        If there is only one guest, then there is no need to aggregate the features.
        In this case, only use the "mean" aggregation.

    .. note::


    .. example::
        >>> from mofdscribe.featurizers.hostguest import HostGuestFeaturizer
        >>> from matminer.featurizers.structure.sites import SiteStatsFingerprint
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("tests/data/structures/Co2O3.cif")
        >>> featurizer = HostGuestFeaturizer(featurizer=SiteStatsFingerprint.from_preset("SOAP_formation_energy"),
            aggregations=("mean", "std", "min", "max"))
    """

    def __init__(
        self,
        featurizer: BaseFeaturizer,
        aggregations: Tuple[str] = ("mean", "std", "min", "max"),
        local_env_method: str = "vesta",
        remove_guests: bool = True,
    ) -> None:
        """
        Construct a new BUFeaturizer.

        Args:
            featurizer (BaseFeaturizer): The featurizer to use.
                Currently, we do not support `MultipleFeaturizer`s.
                Please, instead, use multiple BUFeaturizers.
                If you use a featurizer that is not implemented in mofdscribe
                (e.g. a matminer featurizer), you need to wrap using a method
                that describes on which data objects the featurizer can operate on.
                If you do not do this, we default to assuming that it operates on structures.
            aggregations (Tuple[str]): The aggregations to use.
                Must be one of :py:obj:`ARRAY_AGGREGATORS`.
            local_env_method (str): The method to use for the local environment determination
                (to compute the structure graph). Defaults to "vesta".
            remove_guests (bool): Whether to remove the guests from the structure.
                This is useful if you want to compute features on the host only, independent
                of the guests. Defaults to True.
        """
        self._featurizer = featurizer
        self._aggregations = aggregations
        self._local_env_method = local_env_method
        self._remove_guests = remove_guests
        set_operates_on(self, featurizer)

    def feature_labels(self) -> List[str]:
        labels = []
        base_labels = self._featurizer.feature_labels()
        for label in base_labels:
            labels.append(f"host_{label}")
        for bb in ["guest"]:
            for aggregation in self._aggregations:
                for label in base_labels:
                    labels.append(f"{bb}_{aggregation}_{label}")
        return labels

    def _extract_host_guest(
        self,
        structure: Optional[Union[Structure, IStructure]] = None,
        host_guest: Optional[HostGuest] = None,
    ):
        return _extract_host_guest(
            structure=structure,
            host_guest=host_guest,
            remove_guests=self._remove_guests,
            operates_on=self._operates_on,
            local_env_method=self._local_env_method,
        )

    def fit(
        self,
        structures: Collection[Union[Structure, IStructure]],
        host_guests: Optional[Collection[HostGuest]] = None,
    ) -> None:
        """
        Fit the featurizer to the given structures.

        Args:
            structures (Collection[Union[Structure, IStructure]]): The structures to fit to.
            host_guests (Optional[Collection[HostGuest]]): The host_guests to fit to.
                If you provide this, you must not provide structures.
        """
        all_hosts, all_guests = [], []

        if structures is not None:
            for structure in structures:
                host_guest = self._extract_host_guest(structure=structure)
                all_hosts.append(host_guest.host)
                all_guests.extend(host_guest.guests)

        if host_guests is not None:
            for host_guest in host_guests:
                host_guest = self._extract_host_guest(host_guest=host_guest)
                all_hosts.append(host_guest.host)
                all_guests.extend(host_guest.guests)
        self._featurizer.fit(all_hosts + all_guests)

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

        if host_guest.host is None:
            raise ValueError(
                "Did not find a framework. This is required for the host guest featurizer."
            )

        if len(host_guest.guests) == 0:
            logger.warning(
                "Did not find any guests. Make sure that the guest is not covalently bonded to the host."
            )

        guest_feats = []
        host_feats = self._featurizer.featurize(host_guest.host)

        for guest in host_guest.guests:
            guest_feats.append(self._featurizer.featurize(guest))

        aggregated_feats = []
        for aggregation in self._aggregations:
            aggregated_feats.extend(ARRAY_AGGREGATORS[aggregation](guest_feats, axis=0))

        aggregated_feats = np.array(aggregated_feats)
        return np.concatenate((host_feats, aggregated_feats))

    def citations(self) -> List[str]:
        return self._featurizer.citations()

    def implementors(self) -> List[str]:
        return self._featurizer.implementors()
