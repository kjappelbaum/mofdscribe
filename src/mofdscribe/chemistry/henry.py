# -*- coding: utf-8 -*-
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.raspa.parser import parse
from mofdscribe.utils.raspa.resize_uc import resize_unit_cell
from mofdscribe.utils.raspa.run_raspa import run_raspa

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


def parse_widom(directory: Union[str, Path]) -> dict:
    outputs = glob(os.path.join(directory, "Output", "System_0", "*.data"))
    if len(outputs) != 1:
        raise ValueError("Expected one output file, got {}".format(len(outputs)))
    with open(outputs[0], "r") as handle:
        res = parse(handle.read())
    return [res["Average Henry coefficient"]["Henry"][0], res["HoA_K"]]


class Henry(BaseFeaturizer):
    """
    Computes the Henry coefficient for a given molecule using the RASPA [1]_ program.
    While Henry coefficients are sometimes the targets of ML algorithms, they are sometimes also used as features.
    In fact, they are closely related to the energy histograms.

    """

    def __init__(
        self,
        raspa_dir: Union[str, Path, None] = None,
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
        """

        Args:
            raspa_dir (Union[str, Path, None], optional): Path to the raspa directory (with lib, bin, share) subdirectories.
                If `None` we will look for the `RASPA_DIR` environment variable.
                Defaults to None.
            cycles (int, optional): Number of simulation cycles. Defaults to 5_000.
            temperature (float, optional): Simulation temperature in Kelvin. Defaults to 300.
            cutoff (float, optional): Cutoff for simulation in Angstrom. Defaults to 12.
            mof_ff (str, optional): Name of the forcefield used for the framework. Defaults to "UFF".
            mol_ff (str, optional): Name of the forcefield used for the guest molecule. Defaults to "TraPPE".
            mol_name (str, optional): Name of the guest molecule. Defaults to "CO2".
            tail_corrections (bool, optional): If true, use analytical tail-correction
                for the contribution of the interaction potential after the cutoff. Defaults to True.
            mixing_rule (str, optional): Mixing rule for framework and guest molecule force field. Available options are `Jorgenson` and `Lorentz-Berthelot`. Defaults to "Lorentz-Berthelot".
            shifted (bool, optional): If true, shifts the potential to equal to zero at the cutoff. Defaults to False.
            separate_interactions (bool, optional): If True use framework's force field for framework-molecule interactions.
                Defaults to True.
            run_eqeq (bool, optional): If true, runs EqEq to compute charges. Defaults to True.

        Raises:
            ValueError: _description_
        """
        self.raspa_dir = raspa_dir if raspa_dir else os.environ.get("RASPA_DIR", None)
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
        ucells = " ".format(resize_unit_cell(s, self.cutoff))

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
            r"author = {David Dubbeldam and Sof{'{\i}}a Calero and Donald E. Ellis and Randall Q. Snurr},"
            "title = {{RASPA}: molecular simulation software for adsorption and diffusion in flexible nanoporous materials},"
            "journal = {Molecular Simulation}"
            "}"
        ]
