# -*- coding: utf-8 -*-
from typing import List, Union

from pymatgen.core import Structure, IStructure
import numpy as np
from matminer.featurizers.base import BaseFeaturizer

GRID_INPUT_TEMPLATE = """SimulationType  MakeGrid

Forcefield      {forcefield}

Framework 0
FrameworkName {mof_name}
UnitCells {unit_cells}

UseChargesFromCIFFile   {use_charges}

CutOff                        {cutoff}

ChargeMethod                  Ewald
EwaldPrecision                {ewald_precision}


NumberOfGrids {num_grids}
GridTypes {grid_types}
SpacingVDWGrid {vdw_spacing}
SpacingCoulombGrid {coulomb_spacing}"""


# https://aip.scitation.org/doi/10.1063/5.0050823 proposes to not use equally spaced bins


class EnergyGrid(BaseFeaturizer):
    def __init__(
        self,
        bin_size_vdw: float = 1,
        min_energy_vdw: float = -10,
        max_energy_vdw: float = 0,
        bin_size_el: float = 1,
        min_energy_el: float = -10,
        max_energy_el: float = 0,
    ) -> None:
        super().__init__()

    def fit(self, structures: Union[Structure, IStructure]):
        return self

    def feature_labels(self) -> List[str]:
        return ["EnergyGrid"]

    def featurize(self, s) -> np.array:
        ...

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
