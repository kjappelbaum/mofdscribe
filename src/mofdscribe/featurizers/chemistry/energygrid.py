# -*- coding: utf-8 -*-
"""Featurizer that runs RASPA to calculate energy grids."""
import os
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.histogram import get_rdf
from mofdscribe.featurizers.utils.raspa.resize_uc import resize_unit_cell
from mofdscribe.featurizers.utils.raspa.run_raspa import detect_raspa_dir, run_raspa

from ..utils.extend import operates_on_istructure, operates_on_structure

__all__ = ["EnergyGridHistogram"]
GRID_INPUT_TEMPLATE = """SimulationType  MakeASCIGrid

Forcefield      Local

Framework 0
FrameworkName input
UnitCells {unit_cells}

CutOff                        {cutoff}

NumberOfGrids {num_grids}
GridTypes {grid_types}
SpacingVDWGrid {vdw_spacing}
"""


def parse_energy_grids(directory: Union[str, os.PathLike]) -> dict:
    grids = glob(os.path.join(directory, "ASCI_Grids", "*.grid"))
    energies = {}
    for grid in grids:
        name = os.path.basename(grid).split(".")[0].replace("asci_grid_", "")
        energies[name] = read_ascii_grid(grid)
    return energies


def read_ascii_grid(filename: Union[str, os.PathLike]) -> pd.DataFrame:
    """Read an ASCII grid file into a pandas DataFrame.

    Args:
        filename (Union[str, os.PathLike]): The path to the file to read.

    Returns:
        A pandas DataFrame containing the grid data.
    """
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=["x", "y", "z", "energy", "deriv_x", "deriv_y", "deriv_z"],
        na_values="?",
    )
    df = df.astype(np.float)
    return df


@operates_on_istructure
@operates_on_structure
class EnergyGridHistogram(MOFBaseFeaturizer):
    """Computes the energy grid histograms as originally proposed by Bucior et al. [Bucior2019]_.

    Conventionally, energy grids can be used to speed up molecular simulations.
    The idea is that the interactions between the guest and host are
    pre-computed on a fine grid and then only need to be looked up (instead of
    re-computed all the time).
    Bucior et al. proposed (effectively) a dimensionality reduction of the
    energy grid by making a histogram of the energies. For H2 they used bins of
    a width of 1 kJ mol−1 ranging from −10 kJ mol−1 (attractive) to 0 kJ mol−1.
    For methane, they used bins in 2 kJ mol−1 increments between −26 and 0 kJ
    mol−1, again with a repulsion bin.

    This approach has also been used, for example, `Li et al. (2021)
    <https://aip.scitation.org/doi/10.1063/5.0050823>`_ and [Bucior2021]_.

    References:
        .. [Bucior2021]  Li, Z.; Bucior, B. J.; Chen, H.; Haranczyk, M.; Siepmann, J. I.;
            Snurr, R. Q. Machine Learning Using Host/Guest Energy Histograms to
            Predict Adsorption in Metal–Organic Frameworks: Application to Short
            Alkanes and Xe/Kr Mixtures. J. Chem. Phys. 2021, 155 (1), 014701.
            https://doi.org/10.1063/5.0050823.
    """

    def __init__(
        self,
        raspa_dir: Union[str, os.PathLike, None] = None,
        grid_spacing: float = 1.0,
        bin_size_vdw: float = 1,
        min_energy_vdw: float = -40,
        max_energy_vdw: float = 0,
        cutoff: float = 12,
        mof_ff: str = "UFF",
        mol_ff: str = "TraPPE",
        mol_name: str = "CO2",
        sites: Tuple[str] = ("C_co2",),
        tail_corrections: bool = True,
        mixing_rule: str = "Lorentz-Berthelot",
        shifted: bool = False,
        separate_interactions: bool = True,
        run_eqeq: bool = True,
        primitive: bool = False,
    ):
        """Construct the EnergyGridHistogram class.

        Args:
            raspa_dir (Union[str, Path, None]): Path to the raspa
                directory (with lib, bin, share) subdirectories.
                If `None` we will look for the `RASPA_DIR` environment variable.
                Defaults to None.
            grid_spacing (float): Spacing for the energy grids.
                Bucior et al. (2018) used 1.0 A. Defaults to 1.0.
            bin_size_vdw (float): Size of bins for the energy
                histogram. Defaults to 1.
            min_energy_vdw (float): Minimum
                energy for the energy histogram (defining start of first bin).
                Defaults to -10.
            max_energy_vdw (float): Maximum energy for energy
                histogram (defining last bin).
                Defaults to 0.
            cutoff (float): Cutoff for the Van-der-Waals interaction.
                Defaults to 12.
            mof_ff (str): Name of the forcefield used
                for the framework. Defaults to "UFF".
            mol_ff (str): Name of the forcefield used for the guest molecule.
                Defaults to "TraPPE".
            mol_name (str): Name of the guest molecule.
                Defaults to "CO2".
            sites (Tuple[str]): Name of the Van-der-Waals sites
                for which the energy diagrams are computed.
                Defaults to ("C_co2",).
            tail_corrections (bool): If true, use analytical
                tail-correction for the contribution of the interaction potential after the
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
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to True.

        Raises:
            ValueError: If the `raspa_dir` is not a valid directory.
        """
        self.raspa_dir = raspa_dir if raspa_dir else os.environ.get("RASPA_DIR")
        if self.raspa_dir is None:
            try:
                self.raspa_dir = detect_raspa_dir()
            except ValueError:
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
        self.run_eqeq = run_eqeq
        super().__init__(primitive=primitive)

    def fit_transform(self, structures: List[Union[Structure, IStructure]]):
        ...

    def fit(self, structure: Union[Structure, IStructure]):
        ...

    def _get_grid(self):
        return np.arange(self.min_energy_vdw, self.max_energy_vdw, self.bin_size_vdw)

    def feature_labels(self) -> List[str]:
        grid = self._get_grid()
        labels = []
        for site in self.sites:
            for grid_point in grid:
                labels.append(f"energygridhist_{self.mol_name}_{site}_{grid_point}")
        return labels

    def _featurize(self, s: Union[Structure, IStructure]) -> np.array:
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

        simulation_script = GRID_INPUT_TEMPLATE.format(
            unit_cells=ucells,
            cutoff=self.cutoff,
            num_grids=len(self.sites),
            grid_types=" ".join(self.sites),
            vdw_spacing=self.grid_spacing,
        )
        res = run_raspa(
            s, self.raspa_dir, simulation_script, parameters, parse_energy_grids, self.run_eqeq
        )
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
            "author = {Benjamin J. Bucior and N. Scott Bobbitt and Timur Islamoglu "
            "and Subhadip Goswami and Arun Gopalan and Taner Yildirim and "
            "Omar K. Farha and Neda Bagheri and Randall Q. Snurr},"
            "title = {Energy-based descriptors to rapidly predict hydrogen storage "
            "in metal{\textendash}organic frameworks},"
            r"journal = {Molecular Systems Design {\&}amp$\mathsemicolon$ Engineering}"
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
            r"author = {David Dubbeldam and Sof{'{\i}}a Calero and Donald E. Ellis "
            "and Randall Q. Snurr},"
            "title = {{RASPA}: molecular simulation software for adsorption and "
            "diffusion in flexible nanoporous materials},"
            "journal = {Molecular Simulation}"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka", "David Dubbeldam and RASPA authors"]
