# -*- coding: utf-8 -*-
from functools import lru_cache
from typing import Collection, List, Union

from pymatgen.core import IStructure, Structure


def filter_element(structure: Structure, elements: List[str]) -> Structure:
    """
    Filter a structure by element.
    Args:
        structure (Structure): input structure
        element (str): element to filter
    Returns:
        filtered_structure (Structure): filtered structure
    """

    elements_ = []
    for atom_type in elements:
        if "-" in atom_type:
            elements_.append(atom_type.split("-"))
        else:
            elements_.append([atom_type])
    keep_sites = []
    for site in structure.sites:
        if site.specie.symbol in elements_:
            keep_sites.append(site)
    return Structure.from_sites(keep_sites)


def elements_in_structure(structure: Structure) -> List[str]:
    return list(structure.composition.get_el_amt_dict().keys())


def get_metal_indices(structure: Structure) -> List[int]:
    metal_indices = []
    for i, s in enumerate(structure):
        if s.specie.is_metal:
            metal_indices.append(i)
    return metal_indices


def select_elements(structure: Structure, element: str) -> Structure:
    sites = []
    for site in structure:
        if str(site.specie) == element:
            sites.append(site)

    return Structure.from_sites(sites)


def _not_relevant_structure_indices(structure: Union[Structure, IStructure], indices: Collection):
    return [i for i in range(len(structure)) if i not in indices]
