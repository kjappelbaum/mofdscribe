# -*- coding: utf-8 -*-
import os
from glob import glob
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.histogram import get_rdf
from mofdscribe.utils.raspa.resize_uc import resize_unit_cell
from mofdscribe.utils.raspa.run_raspa import run_raspa

GRID_INPUT_TEMPLATE = """SimulationType  MakeASCIGrid

Forcefield      {forcefield}

Framework 0
FrameworkName input
UnitCells {unit_cells}

CutOff                        {cutoff}

NumberOfGrids {num_grids}
GridTypes {grid_types}
SpacingVDWGrid {vdw_spacing}
"""


# https://aip.scitation.org/doi/10.1063/5.0050823 proposes to not use equally spaced bins


# Bucior used 1 A spacing, the LJ site for H2 and no Coulomb grid
# set raspa_dir to root dir of conda env, e.g., /Users/leopold/Applications/miniconda3/envs/simulations


def parse_energy_grids(directory: Union[str, Path]) -> dict:
    grids = glob(os.path.join(directory, "*.grid"))
    energies = {}
    for grid in grids:
        name = os.path.basename(grid).split(".")[0].replace("asci_grid_", "")
        energies[name] = read_ascii_grid(grid)
    return energies


def read_ascii_grid(filename: str) -> pd.DataFrame:
    """
    Read an ASCII grid file into a pandas DataFrame.

    Args:
        filename: The path to the file to read.

    Returns:
        A pandas DataFrame containing the grid data.
    """
    df = pd.read_csv(
        filename,
        sep="\s+",
        header=None,
        names=["x", "y", "z", "energy", "deriv_x", "deriv_y", "deriv_z"],
    )
    df = df.replace("?", np.nan)
    df = df.astype(np.float)
    return df


class EnergyGridHistogram(BaseFeaturizer):
    """Computes the energy grid histograms as originally proposed by Bucior et al."""

    def __init__(
        self,
        raspa_dir: Union[str, Path, None] = None,
        grid_spacing: float = 1.0,
        bin_size_vdw: float = 1,
        min_energy_vdw: float = -10,
        max_energy_vdw: float = 0,
        cutoff: float = 12,
        mof_ff: str = "UFF",
        mol_ff: str = "TraPPE",
        mol_name: str = "CO2",
        sites: List[str] = ["C_co2"],
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
        self.grid_spacing = grid_spacing
        self.bin_size_vdw = bin_size_vdw
        self.min_energy_vdw = min_energy_vdw
        self.max_energy_vdw = max_energy_vdw
        self.cutoff = cutoff
        self.mof_ff = mof_ff
        self.mol_ff = mol_ff
        self.mol_name = mol_name
        self.sites = sites
        self.tail_corrections = tail_corrections
        self.mixing_rule = mixing_rule
        self.shifted = shifted
        self.separate_interactions = separate_interactions

    def fit_transform(self, structures: List[Union[Structure, IStructure]]):
        ...

    def fit(self, structure: Union[Structure, IStructure]):
        return self

    def _get_grid(self):
        return np.arange(self.min_energy_vdw, self.max_energy_vdw, self.bin_size_vdw)

    def feature_labels(self) -> List[str]:
        grid = self._get_grid()
        labels = []
        for site in self.sites:
            for grid_point in grid:
                labels.append(f"{site}_{grid_point}")
        return labels

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

        simulation_script = GRID_INPUT_TEMPLATE.format(
            forcefield=self.mof_ff,
            unit_cells=ucells,
            cutoff=self.cutoff,
            num_grids=len(self.sites),
            grid_types=" ".join(self.sites),
            vdw_spacing=self.grid_spacing,
        )
        res = run_raspa(s, self.raspa_dir, simulation_script, parameters, parse_energy_grids)
        output = []
        for _, v in res.items():
            output.append(
                get_rdf(
                    v["energy"].values,
                    self.min_energy_vdw,
                    self.max_energy_vdw,
                    self.bin_size_vdw,
                    None,
                    None,
                    normalized=False,
                )
            )
        return np.concatenate(output)

    def citations(self) -> List[str]:
        return [
            "@article{Bucior2019,"
            "doi = {10.1039/c8me00050f},"
            "url = {https://doi.org/10.1039/c8me00050f},"
            "year = {2019},"
            "publisher = {Royal Society of Chemistry ({RSC})},"
            "volume = {4},"
            "number = {1},"
            "pages = {162--174},"
            "author = {Benjamin J. Bucior and N. Scott Bobbitt and Timur Islamoglu and Subhadip Goswami and Arun Gopalan and Taner Yildirim and Omar K. Farha and Neda Bagheri and Randall Q. Snurr},"
            "title = {Energy-based descriptors to rapidly predict hydrogen storage in metal{\textendash}organic frameworks},"
            "journal = {Molecular Systems Design {\&}amp$\mathsemicolon$ Engineering}"
            "}",
            "@article{Dubbeldam2015,"
            "doi = {10.1080/08927022.2015.1010082},"
            "url = {https://doi.org/10.1080/08927022.2015.1010082},"
            "year = {2015},"
            "month = feb,"
            "publisher = {Informa {UK} Limited},"
            "volume = {42},"
            "number = {2},"
            "pages = {81--101},"
            "author = {David Dubbeldam and Sof{'{\i}}a Calero and Donald E. Ellis and Randall Q. Snurr},"
            "title = {{RASPA}: molecular simulation software for adsorption and diffusion in flexible nanoporous materials},"
            "journal = {Molecular Simulation}"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
