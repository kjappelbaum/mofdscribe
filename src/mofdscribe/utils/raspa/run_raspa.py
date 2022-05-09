# -*- coding: utf-8 -*-
RUN_SCRIPT = """#! /bin/sh -f
export DYLD_LIBRARY_PATH=RASPA_DIR/lib
export LD_LIBRARY_PATH=RASPA_DIR/lib
RASPA_DIR/bin/simulate
"""

import os
import subprocess
from tempfile import TemporaryDirectory

from .ff_builder import ff_builder


def run_raspa(structure, raspa_dir, simulation_script, ff_params, parser, run_eqeq: bool = False):
    ff_results = ff_builder(ff_params)
    with TemporaryDirectory() as tempdir:
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
                _ = subprocess.run(
                    ["eqeq", "input.cif", "-o", "input.cif"],
                    universal_newlines=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    cwd=tempdir,
                )
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Error running EqEq. Output: {e.output}  stderr: {e.stderr}")

        try:
            _ = subprocess.run(
                ["sh", "run.sh"],
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                cwd=tempdir,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error running RASPA. Output: {e.output}  stderr: {e.stderr}")

        results = parser(os.path.join(tempdir))

    return results
