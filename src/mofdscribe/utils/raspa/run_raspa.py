# -*- coding: utf-8 -*-
RUN_SCRIPT = """#! /bin/sh -f
export DYLD_LIBRARY_PATH=RASPA_DIR/lib
export LD_LIBRARY_PATH=RASPA_DIR/lib
RASPA_DIR/bin/simulate
"""

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

from pymatgen.core import IStructure, Structure

from mofdscribe.utils.eqeq import get_eqeq_charges

from .ff_builder import ff_builder
from ..tempdir import TEMPDIR


def call_eqeq(structure, filename):
    if isinstance(structure, Structure):
        structure = IStructure.from_sites(structure)
    s, _ = get_eqeq_charges(structure)
    # This is a weird hack because recent versions of ASE changed how they write
    # symmetry in CIFs. One representative issue is https://github.com/lsmo-epfl/curated-cofs-submission/issues/17
    s = s.replace("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M")
    with open(filename, "w") as handle:
        handle.write(s)


def run_raspa(
    structure: Union[Structure, IStructure],
    raspa_dir: Union[str, Path],
    simulation_script,
    ff_params,
    parser,
    run_eqeq: bool = False,
):
    ff_results = ff_builder(ff_params)
    with TemporaryDirectory(dir=TEMPDIR) as tempdir:
        for k, v in ff_results.items():
            with open(
                os.path.join(tempdir, k.replace("_def", ".def").replace("molecule_", "")), "w"
            ) as handle:
                handle.write(v)

        with open(os.path.join(tempdir, "simulation.input"), "w") as handle:
            handle.write(simulation_script)

        with open(os.path.join(tempdir, "run.sh"), "w") as handle:
            run_template = RUN_SCRIPT.replace("RASPA_DIR", raspa_dir)
            handle.write(run_template)

        structure.to("cif", os.path.join(tempdir, "input.cif"))
        if run_eqeq:
            try:
                call_eqeq(structure, os.path.join(tempdir, "input.cif"))
            except Exception as e:
                raise ValueError(f"Error running EqEq. Output: {e}")

        try:
            _ = subprocess.run(
                ["sh", "run.sh"],
                universal_newlines=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                cwd=tempdir,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error running RASPA. Output: {e.output}  stderr: {e.stderr}")

        results = parser(os.path.join(tempdir))

    return results
