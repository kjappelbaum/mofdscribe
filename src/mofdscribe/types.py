"""Types that are reused across the mofdscribe pacakge."""
from pathlib import Path
from typing import Union

from pymatgen.core.structure import IStructure, Structure
from typing_extensions import TypeAlias

PathType: TypeAlias = Union[str, Path]
StructureIStructureType: TypeAlias = Union[IStructure, Structure]
