from pymatgen.core import Structure, IStructure
from typing import List
from functools import lru_cache


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
