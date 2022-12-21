from functools import lru_cache
from typing import List, Union

from pymatgen.core import IStructure, Structure


@lru_cache(maxsize=None)
def _fragment_cached(structure):
    from moffragmentor import MOF

    mof = MOF.from_structure(structure)
    fragments = mof.fragment()
    return fragments


def fragment(structure: Union[Structure, IStructure]) -> List[Structure]:
    if isinstance(structure, Structure):
        return _fragment_cached(IStructure.from_sites(structure.sites))
    elif isinstance(structure, IStructure):
        return _fragment_cached(structure)
    else:
        raise TypeError(f"Expected Structure or IStructure, got {type(structure)}")
