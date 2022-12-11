# -*- coding: utf-8 -*-
"""Helper functions for the host-guest featurizers."""
from typing import List

from pymatgen.core import Structure


def remove_guests_from_structure(structure, guest_indices: List[List[int]]) -> Structure:
    """
    Remove guests from a structure.

    Args:
        structure (Structure): The structure to remove guests from.
        guest_indices (List[List[int]]): The indices of the guests.

    Returns:
        Structure: The structure without guests.
    """
    flattened_guest_indices = sum(guest_indices, [])
    ok_sites = []
    for i, site in enumerate(structure):
        if i not in flattened_guest_indices:
            ok_sites.append(site)

    return Structure.from_sites(ok_sites)
