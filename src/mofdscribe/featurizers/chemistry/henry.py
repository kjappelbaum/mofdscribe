# -*- coding: utf-8 -*-
"""Featurizer that runs RASPA to calculate the Henry coefficient."""
import os
from glob import glob
from typing import List, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.utils.extend import operates_on_istructure, operates_on_structure
from mofdscribe.featurizers.utils.raspa.parser import parse
from mofdscribe.featurizers.utils.raspa.resize_uc import resize_unit_cell
from mofdscribe.featurizers.utils.raspa.run_raspa import run_raspa

__all__ = ["Henry"]

WIDOM_INPUT_TEMPLATE = """SimulationType  MonteCarlo

Forcefield      Local
NumberOfCycles                {cycles}
PrintEvery {print_every}
NumberOfInitializationCycles  0

Framework 0
FrameworkName input
UnitCells {unit_cells}
UseChargesFromCIFFile yes

CutOff                        {cutoff}

ExternalTemperature {temp}


Component 0 MoleculeName             {molname}
            MoleculeDefinition       Local
            WidomProbability         1.0
            CreateNumberOfMolecules  0
"""


def parse_widom(directory: Union[str, os.PathLike]) -> dict:
    """Parse the widom output files in the given directory."""
    outputs = glob(os.path.join(directory, "Output", "System_0", "*.data"))
    if len(outputs) != 1:
        raise ValueError("Expected one output file, got {}".format(len(outputs)))
    with open(outputs[0], "r") as handle:
        res = parse(handle.read())
    return [res["Average Henry coefficient"]["Henry"][0], res["HoA_K"]]


@operates_on_structure
@operates_on_istructure
class Henry(BaseFeaturizer):
    """Computes the Henry coefficient for a given molecule using the RASPA [1]_ program.

    While Henry coefficients are sometimes the targets of ML
    algorithms, they are sometimes also used as features. In fact, they are
    closely related to the energy histograms.
    """

    def __init__(
        self,
        raspa_dir: Union[str, os.PathLike, None] = None,
        cycles: int = 5_000,
        temperature: float = 300,
        cutoff: float = 12,
        mof_ff: str = "UFF",
        mol_ff: str = "TraPPE",
        mol_name: str = "CO2",
        tail_corrections: bool = True,
        mixing_rule: str = "Lorentz-Berthelot",
        shifted: bool = False,
        separate_interactions: bool = True,
        run_eqeq: bool = True,
    ):
        """Initialize the featurizer.

        Args:
            raspa_dir (Union[str, PathLike, None]): Path to the raspa
                directory (with lib, bin, share) subdirectories.
                If `None` we will look for the `RASPA_DIR` environment variable.
                Defaults to None.
            cycles (int): Number of simulation cycles.
                Defaults to 5_000.
            temperature (float): Simulation temperature in
                Kelvin. Defaults to 300.
            cutoff (float): Cutoff for simulation in Angstrom.
                Defaults to 12.
            mof_ff (str): Name of the forcefield used for the framework.
                Defaults to "UFF".
            mol_ff (str): Name of the forcefield used for the guest molecule.
                Defaults to "TraPPE".
            mol_name (str): Name of the guest molecule. Defaults to "CO2".
            tail_corrections (bool): If true, use analytical tail-correction
                for the contribution of the interaction potential after the
                cutoff. Defaults to True.
            mixing_rule (str): Mixing rule for framework and guest
                molecule force field. Available options are `Jorgenson` and
                `Lorentz-Berthelot`. Defaults to "Lorentz-Berthelot".
            shifted (bool): If true, shifts the potential to equal to zero at the
                cutoff. Defaults to False.
            separate_interactions (bool): If True use framework's force field
                for framework-molecule interactions.
                Defaults to True.
            run_eqeq (bool): If true, runs EqEq to compute charges.
                Defaults to True.

        Raises:
            ValueError: If the `RASPA_DIR` environment variable is not set.
        """
        self.raspa_dir = raspa_dir if raspa_dir else os.environ.get("RASPA_DIR")
        if self.raspa_dir is None:
            raise ValueError(
                "Please set the RASPA_DIR environment variable or provide the path for the class initialization."
            )
        self.cycles = cycles
        self.cutoff = cutoff
        self.mof_ff = mof_ff
        self.mol_ff = mol_ff
        self.mol_name = mol_name
        self.tail_corrections = tail_corrections
        self.mixing_rule = mixing_rule
        self.shifted = shifted
        self.separate_interactions = separate_interactions
        self.temperature = temperature
        self.run_eqeq = run_eqeq

    def featurize(self, s: Union[Structure, IStructure]) -> np.array:
        ff_molecules = {self.mol_name: self.mol_ff}

        parameters = {
            "ff_framework": self.mof_ff,
            "ff_molecules": ff_molecules,
            "shifted": self.shifted,
            "tail_corrections": self.tail_corrections,
            "mixing_rule": self.mixing_rule,
            "separate_interactions": self.separate_interactions,
        }
        replicas = resize_unit_cell(s, self.cutoff)
        ucells = f"{replicas[0]} {replicas[1]} {replicas[2]}"
        simulation_script = WIDOM_INPUT_TEMPLATE.format(
            cycles=self.cycles,
            unit_cells=ucells,
            cutoff=self.cutoff,
            print_every=self.cycles // 10,
            molname=self.mol_name,
            temp=self.temperature,
        )
        res = run_raspa(
            s,
            self.raspa_dir,
            simulation_script,
            parameters,
            parse_widom,
            self.run_eqeq,
        )
        return np.array(res)

    def feature_labels(self) -> List[str]:
        return [
            f"henry_coefficient_{self.mol_name}_{self.temperature}_mol/kg/Pa",
            f"heat_of_adsorption_{self.mol_name}_{self.temperature}_K",
        ]

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]

    def citations(self) -> List[str]:
        return [
            "@article{Dubbeldam2015,"
            "doi = {10.1080/08927022.2015.1010082},"
            "url = {https://doi.org/10.1080/08927022.2015.1010082},"
            "year = {2015},"
            "month = feb,"
            "publisher = {Informa {UK} Limited},"
            "volume = {42},"
            "number = {2},"
            "pages = {81--101},"
            r"author = {David Dubbeldam and Sof{'{\i}}a Calero and "
            "Donald E. Ellis and Randall Q. Snurr},"
            "title = {{RASPA}: molecular simulation software for adsorption "
            "and diffusion in flexible nanoporous materials},"
            "journal = {Molecular Simulation}"
            "}"
        ]
