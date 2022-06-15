# -*- coding: utf-8 -*-
from typing import Union
import os

from pymatgen.core import IStructure, Structure


class MOF:
    def __init__(self, structure: Structure) -> None:
        self.structure = IStructure.from_site(structure.sites)

    @classmethod
    def from_cif(self, path: Union[str, os.PathLike]) -> "MOF":
        s = IStructure.from_file(path)
        return MOF(s)
