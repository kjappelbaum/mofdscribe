from pathlib import Path
from typing import Union

from pymatgen.core import IStructure, Structure


class MOF:
    def __init__(self, structure: Structure) -> None:
        self.structure = IStructure.from_site(structure.sites)

    @classmethod
    def from_cif(self, path: Union[str, Path]):
        s = IStructure.from_file(path)
        return MOF(s)
