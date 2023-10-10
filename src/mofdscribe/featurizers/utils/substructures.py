# -*- coding: utf-8 -*-
"""Extract substructures (e.g. of certain element types)."""
from typing import Collection, List, Union

from pymatgen.core import IMolecule, IStructure, Molecule, Structure


def filter_element(
    structure: Union[Structure, IStructure, Molecule, IMolecule],
    elements: List[str],
    return_indices=False,
) -> Structure:
    """Filter a structure by element.

    Args:
        structure (Union[Structure, IStructure, Molecule, IMolecule]): input structure
        elements (str): element to filter
        return_indices (bool): whether to return the indices of the filtered sites

    Returns:
        filtered_structure (Structure): filtered structure
    """
    elements_ = []
    for atom_type in elements:
        if "-" in atom_type:
            elements_.extend(atom_type.split("-"))
        else:
            elements_.append(atom_type)
    keep_sites = []
    keep_indices = []
    for i, site in enumerate(structure.sites):
        if site.specie.symbol in elements_:
            keep_sites.append(site)
            keep_indices.append(i)
    if len(keep_sites) == 0:
        return None

    if return_indices:
        return keep_indices
    input_is_structure = isinstance(structure, (Structure, IStructure))
    if input_is_structure:
        return Structure.from_sites(keep_sites)
    else:  # input is molecule or IMolecule
        return Molecule.from_sites(keep_sites)


def elements_in_structure(structure: Structure) -> List[str]:
    """Get all elements symbols in a structure."""
    return list(structure.composition.get_el_amt_dict().keys())


def get_metal_indices(structure: Structure) -> List[int]:
    """Get the indices of metals in the structure."""
    metal_indices = []
    for i, s in enumerate(structure):
        if s.specie.is_metal:
            metal_indices.append(i)
    return metal_indices


def select_elements(structure: Structure, element: str) -> Structure:
    """Return a structure with only the given element."""
    sites = []
    for site in structure:
        if str(site.specie) == element:
            sites.append(site)

    return Structure.from_sites(sites)


def _not_relevant_structure_indices(structure: Union[Structure, IStructure], indices: Collection):
    """Return the indices of the sites that are not included in the provided indices."""
    return [i for i in range(len(structure)) if i not in indices]
