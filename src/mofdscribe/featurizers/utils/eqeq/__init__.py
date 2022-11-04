# -*- coding: utf-8 -*-
"""Run eqeq."""
import contextlib
import os
from functools import lru_cache
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import List, Tuple

from ase.io.cif import write_cif
from pyeqeq import run_on_cif
from pymatgen.core import IStructure
from pymatgen.io.ase import AseAtomsAdaptor


@lru_cache(maxsize=32)
def get_eqeq_charges(structure: IStructure) -> Tuple[str, List[float]]:
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), NamedTemporaryFile(
        "w", suffix=".cif"
    ) as f:
        structure.to(filename=f.name, fmt="cif")
        charges = run_on_cif(f.name)

    atoms = AseAtomsAdaptor().get_atoms(structure)
    charge_dict = dict(zip(range(len(atoms)), charges))
    d = {"atom_site_charge": {0: charge_dict}}

    bo = BytesIO()

    write_cif(bo, atoms, loop_keys=d)

    return bo.getvalue().decode("utf-8"), charges
