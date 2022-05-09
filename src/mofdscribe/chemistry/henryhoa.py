import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.raspa.resize_uc import resize_unit_cell
from mofdscribe.utils.raspa.run_raspa import run_raspa

__all__ = ["HenryHoA"]

WIDOM_INPUT_TEMPLATE = """SimulationType  MonteCarlo

Forcefield      {forcefield}
NumberOfCycles                {cycles}
NumberOfInitializationCycles  0

Framework 0
FrameworkName input
UnitCells {unit_cells}

CutOff                        {cutoff}

ExternalTemperature {temp}


Component 0 MoleculeName             {molname}
            MoleculeDefinition       {moldef}
            WidomProbability         1.0
            CreateNumberOfMolecules  0
"""


def parse_widom(directory: Union[str, Path]) -> dict:
    ...


class HenryHoA(BaseFeaturizer):
    """
    Computes the Henry coefficient and heat of adsorption for a given molecule using the RASPA [1]_ program.
    While Henry coefficients are sometimes the targets of ML algorithms, they are sometimes also used as features.
    In fact, they are closely related to the energy histograms.

    """

    def __init__(
        self,
        raspa_dir: Union[str, Path, None] = None,
        cutoff: float = 12,
        mof_ff: str = "UFF",
        mol_ff: str = "TraPPE",
        mol_name: str = "CO2",
        tail_corrections: bool = True,
        mixing_rule: str = "Lorentz-Berthelot",
        shifted: bool = False,
        separate_interactions: bool = True,
    ):
        self.raspa_dir = raspa_dir if raspa_dir else os.environ.get("RASPA_DIR", None)
        if self.raspa_dir is None:
            raise ValueError(
                "Please set the RASPA_DIR environment variable or provide the path for the class initialization."
            )
        self.cutoff = cutoff
        self.mof_ff = mof_ff
        self.mol_ff = mol_ff
        self.mol_name = mol_name
        self.tail_corrections = tail_corrections
        self.mixing_rule = mixing_rule
        self.shifted = shifted
        self.separate_interactions = separate_interactions

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
            forcefield=self.mof_ff,
            unit_cells=ucells,
            cutoff=self.cutoff,
            num_grids=len(self.sites),
            grid_types=" ".join(self.sites),
            vdw_spacing=self.grid_spacing,
        )
        res = run_raspa(s, self.raspa_dir, simulation_script, parameters, parse_widom)

    def feature_labels(self) -> List[str]:
        ...

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
