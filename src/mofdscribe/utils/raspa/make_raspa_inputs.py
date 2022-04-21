RUN_SCRIPT = """"#! /bin/sh -f
export RASPA_DIR=${RASPA_DIR}/RASPA/simulations/
export DYLD_LIBRARY_PATH=${RASPA_DIR}/lib
export LD_LIBRARY_PATH=${RASPA_DIR}/lib
$RASPA_DIR/bin/simulate $1
"""


def run_raspa(raspa_dir, simulation_input, ff_mixing, pseduo_atoms, mol_files, parser):
    ...
