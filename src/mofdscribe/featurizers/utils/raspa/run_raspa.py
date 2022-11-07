# -*- coding: utf-8 -*-
"""Helper functions to submit RASPA simulations."""

import os
import subprocess
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import Callable, Union

from loguru import logger
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.utils.eqeq import get_eqeq_charges

from .ff_builder import ff_builder
from ..tempdir import TEMPDIR

RUN_SCRIPT = """#! /bin/sh -f
export DYLD_LIBRARY_PATH=RASPA_DIR/lib
export LD_LIBRARY_PATH=RASPA_DIR/lib
RASPA_DIR/bin/simulate
"""


def detect_raspa_dir():
    """Detect the RASPA directory."""
    raspa_dir = which("simulate")
    if raspa_dir is None:
        raise ValueError("RASPA not found.")
    else:
        logger.info(
            "RASPA_DIR not set in environment and input." " Attempting automatic detection."
        )
        p = Path(which("simulate"))
        return os.path.join(*p.parts[:-2])


def call_eqeq(structure: Union[Structure, IStructure], filename: Union[str, os.PathLike]) -> None:
    """Call EqEq to get a structure file `filename` with the charges."""
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
    raspa_dir: Union[str, os.PathLike],
    simulation_script: str,
    ff_params: dict,
    parser: Callable,
    run_eqeq: bool = False,
):
    """Submit a simulation to RASPA.

    Args:
        structure (Union[Structure, IStructure]): Input structure for the framework.
        raspa_dir (Union[str, os.PathLike]): Used for the `RASPA_DIR` environment variable.
        simulation_script (str): RASPA input file.
        ff_params (dict): settings for the force field builder.
        parser (Callable): function that takes the simulation directory as input
            and returns the output.
        run_eqeq (bool): If true, runs eqeq before submitting the RASPA simulations.
            Defaults to False.

    Raises:
        ValueError: In case the simulation fails.

    Returns:
        output created by the parser function.
    """
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

        structure.to(filename=os.path.join(tempdir, "input.cif"), fmt="cif")
        if run_eqeq:
            try:
                call_eqeq(structure, os.path.join(tempdir, "input.cif"))
            except Exception as e:
                raise ValueError(f"Error running EqEq. Output: {e}")

        try:
            _ = subprocess.run(  # noqa: S607, nosec
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
