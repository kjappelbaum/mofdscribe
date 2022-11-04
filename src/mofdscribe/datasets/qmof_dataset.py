# -*- coding: utf-8 -*-
"""Subset of the QMOF dataset."""
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mofdscribe.constants import MOFDSCRIBE_PYSTOW_MODULE
from mofdscribe.datasets.checks import check_all_file_exists, length_check
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.datasets.utils import compress_dataset

__all__ = ["QMOFDataset"]


class QMOFDataset(AbstractStructureDataset):
    """
    Exposes the QMOF dataset by Rosen et al. [Rosen2021]_ [Rosen2022]_ .

    Currently based on v14 of the QMOF dataset.

    To reduce the risk of data leakage, we (by default) also only keep one representative
    structure for a "base refcode" (i.e. the first five letters of a refcode).
    For instance, the base refcode for IGAHED001 is IGAHED. Structures with same
    base refcode but different refcodes are often different refinements, or measurements
    at different temperatures and hence chemically quite similar.
    For instance, in the QMOF dataset `the basecode BOJKAM appears
    four times
    <https://materialsproject.org/mofs?_sort=data.lcd.value&data__csdRefcode__contains=BOJKAM>`_.
    Additionally, we (by default) only keep one structure per "structure hash" which
    is an approximate graph-isomoprhism check, assuming the VESTA bond thresholds
    for the derivation of the structure graph.

    Note that Rosen et al. already performed some deduplication using the pymatgen `StructureMatcher`.
    Our de-duplication is a bit more aggressive, and might be too aggressive in some cases.

    .. warning::
        Even though we performed some basic sanity checks and Rosen et al.
        included checks to ensure high-fidelity structures, there might
        still be some structures that are not chemically reasonable.
        Also, even though we only keep one structure per base refcode, there is still
        potential for data leakge. We urge users to still drop duplicates (or close neighbors)
        after featurization.

    This dataset is available in different flavors:

    * ``"all"``: the full dataset, all original QMOF structures for which we
        could compute features and hashes
    * ``"csd"``:  the subset which comes from the CSD and for which
        we could retrieve publication years.
    * ``"gcmc"``: the subset for which we performed grand canonical Monte Carlo
        simulations
    * ``"gcmc-csd"``: the subset for which we performed grand canonical Monte Carlo
        simulations and for which we could retrieve publication years.

    Currently, we expose the following labels:

        * outputs.pbe.energy_total
        * outputs.pbe.energy_vdw
        * outputs.pbe.energy_elec
        * outputs.pbe.net_magmom
        * outputs.pbe.bandgap
        * outputs.pbe.cbm
        * outputs.pbe.vbm
        * outputs.pbe.directgap
        * outputs.pbe.bandgap_spins
        * outputs.pbe.cbm_spins
        * outputs.pbe.vbm_spins
        * outputs.pbe.directgap_spins
        * outputs.hle17.energy_total
        * outputs.hle17.energy_vdw
        * outputs.hle17.energy_elec
        * outputs.hle17.net_magmom
        * outputs.hle17.bandgap
        * outputs.hle17.cbm
        * outputs.hle17.vbm
        * outputs.hle17.directgap
        * outputs.hle17.bandgap_spins
        * outputs.hle17.cbm_spins
        * outputs.hle17.vbm_spins
        * outputs.hle17.directgap_spins
        * outputs.hse06_10hf.energy_total
        * outputs.hse06_10hf.energy_vdw
        * outputs.hse06_10hf.energy_elec
        * outputs.hse06_10hf.net_magmom
        * outputs.hse06_10hf.bandgap
        * outputs.hse06_10hf.cbm
        * outputs.hse06_10hf.vbm
        * outputs.hse06_10hf.directgap
        * outputs.hse06_10hf.bandgap_spins
        * outputs.hse06_10hf.cbm_spins
        * outputs.hse06_10hf.vbm_spins
        * outputs.hse06_10hf.directgap_spins
        * outputs.hse06.energy_total
        * outputs.hse06.energy_vdw
        * outputs.hse06.energy_elec
        * outputs.hse06.net_magmom
        * outputs.hse06.bandgap
        * outputs.hse06.cbm
        * outputs.hse06.vbm
        * outputs.hse06.directgap
        * outputs.hse06.bandgap_spins
        * outputs.hse06.cbm_spins
        * outputs.hse06.vbm_spins
        * outputs.hse06.directgap_spins
        * outputs.CO2_Henry_coefficient
        * outputs.CO2_adsorption_energy
        * outputs.N2_Henry_coefficient
        * outputs.N2_adsorption_energy
        * outputs.CO2_parasitic_energy_(coal)
        * outputs.Gravimetric_working_capacity_(coal)
        * outputs.Volumetric_working_capacity_(coal)
        * outputs.CO2_parasitic_energy_(nat_gas)
        * outputs.Gravimetric_working_capacity_(nat_gas)
        * outputs.Volumetric_working_capacity_(nat_gas)
        * outputs.Final_CO2_purity_(nat_gas)
        * outputs.CH4_Henry_coefficient
        * outputs.CH4_adsorption_energy
        * outputs.Enthalphy_of_Adsorption__at__58_bar,_298K
        * outputs.Enthalphy_of_Adsorption__at__65bar--298K
        * outputs.Working_capacity_vol_(58-65bar--298K)
        * outputs.Working_capacity_mol_(58-65bar--298K)
        * outputs.Working_capacity_fract_(58-65bar--298K)
        * outputs.Working_capacity_wt%_(58-65bar--298K)
        * outputs.O2_Henry_coefficient
        * outputs.O2_adsorption_energy
        * outputs.Enthalphy_of_Adsorption__at__5_bar,_298K
        * outputs.Enthalphy_of_Adsorption__at__140bar--298K
        * outputs.Working_capacity_vol_(5-140bar--298K)
        * outputs.Working_capacity_mol_(5-140bar--298K)
        * outputs.Working_capacity_fract_(5-140bar--298K)
        * outputs.Working_capacity_wt%_(5-140bar--298K)
        * outputs.Xe_Henry_coefficient
        * outputs.Xe_adsorption_energy
        * outputs.Kr_Henry_coefficient
        * outputs.Kr_adsorption_energy
        * outputs.Xe--Kr_selectivity__at__298K
        * outputs.Working_capacity_g--L_(5-100bar--298-198K)
        * outputs.Working_capacity_g--L_(5-100bar--77K)
        * outputs.Working_capacity_g--L_(1-100bar--77K)
        * outputs.Working_capacity_wt%_(5-100bar--298-198K)
        * outputs.Working_capacity_wt%_(5-100bar--77K)
        * outputs.Working_capacity_wt%_(1-100bar--77K)
        * outputs.H2S_Henry_coefficient
        * outputs.H2S_adsorption_energy
        * outputs.H2O_Henry_coefficient
        * outputs.H2O_adsorption_energy
        * outputs.H2S--H2O_selectivity__at__298K
        * outputs.CH4--N2_selectivity__at__298K

    Note that many of the gas adsorption data are :py:obj:`numpy.nan` because the pores
    are not accessible to the guest molecules. Depending on your application you might want
    to fill them with zeros or drop them.

    .. warning::

        The class will load almost 1GB of data into memory.

    .. warning::

        By default, the values will be sorted by the PBE total energy

    References:
        .. [Rosen2021] `Rosen, A. S.; Iyer, S. M.; Ray, D.; Yao, Z.; Aspuru-Guzik, A.; Gagliardi, L.;
            Notestein, J. M.; Snurr, R. Q. Machine Learning the Quantum-Chemical Properties
            of Metal–Organic Frameworks for Accelerated Materials Discovery.
            Matter 2021, 4 (5), 1578–1597. <https://doi.org/10.1016/j.matt.2021.02.015>`_

        .. [Rosen2022] `Rosen, A. S.; Fung, V.; Huck, P.; O'Donnell, C. T.; Horton, M. K.; Truhlar, D. G.;
            Persson, K. A.; Notestein, J. M.; Snurr, R. Q.
            High-Throughput Predictions of Metal–Organic Framework Electronic Properties:
            Theoretical Challenges, Graph Neural Networks, and Data Exploration.
            npj Computational Materials, 8, 112.
            <https://doi.org/10.1038/s41524-022-00796-6>`_

    """

    # we expect this len for the full dataset
    _files = {
        "v0.0.1": {
            "df": "https://zenodo.org/record/7031397/files/data.json?download=1",
            "structures": "https://zenodo.org/record/7031397/files/structures.tar.gz?download=1",
            "expected_length": 15042,
            "flavors": {
                "all": 15042,
                "csd": 6311,
                "gcmc": 5321,
                "csd-gcmc": 2295,
            },
        }
    }

    def __init__(
        self,
        version: str = "v0.0.1",
        flavor: str = "all",
        drop_basename_duplicates: bool = True,
        drop_graph_duplicates: bool = True,
        subset: Optional[Iterable[int]] = None,
        drop_nan: bool = False,
    ):
        """Construct an instance of the QMOF dataset.

        Args:
            version (str): version number to use.
                Defaults to "v0.0.1".
            flavor (str): flavor of the dataset to use.
                Accepted values are "all", "csd", "gcmc", and "csd-gcmc".
                Defaults to "all".
            drop_basename_duplicates (bool): If True, keep only one structure
                per CSD basename. Defaults to True.
            drop_graph_duplicates (bool): If True, keep only one structure
                per decorated graph hash. Defaults to True.
            subset (Optional[Iterable[int]]): indices of the structures to include.
                This is useful for subsampling the dataset. Defaults to None.
            drop_nan (bool): If True, drop rows with NaN values in features or hashes.
                Defaults to False.

        Raises:
            ValueError: If the provided version number is not available.
        """
        self._drop_basename_duplicates = drop_basename_duplicates
        self._drop_nan = drop_nan
        self._drop_graph_duplicates = drop_graph_duplicates
        self._flavor = flavor
        if version not in self._files:
            raise ValueError(
                f"Version {version} not available. Available versions: {list(self._files.keys())}"
            )
        if flavor not in self._files[version]["flavors"]:
            raise ValueError(
                f"Flavor {flavor} not available. Available flavors: {list(self._files[version]['flavors'].keys())}"
            )
        self.version = version

        # download the data for the largest set ("all")
        self._structure_dir = MOFDSCRIBE_PYSTOW_MODULE.ensure_untar(
            "QMOF",
            self.version,
            name="structures.tar.gz",
            url=self._files[version]["structures"],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                "QMOF", self.version, name="data.json", url=self._files[version]["df"]
            )
        ).reset_index(drop=True)
        compress_dataset(self._df)
        length_check(self._df, self._files[version]["expected_length"])

        # we sort by the PBE energy to make sure we keep always the lowest in energy
        self._df = self._df.sort_values(by="outputs.pbe.energy_total")
        if drop_basename_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["info.basename"], keep="first")
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate basenames. New length {len(self._df)}"
            )
        if drop_graph_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["info.decorated_graph_hash"], keep="first")
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate graphs. New length {len(self._df)}"
            )

        # select by flavor
        self._df = self._df[self._df[f"flavor.{self._flavor}"]]

        self._df = self._df.reset_index(drop=True)

        if drop_nan:
            self._df.dropna(
                subset=[c for c in self._df.columns if c.startswith("features.")]
                + [c for c in self._df.columns if c.startswith("info.")],
                inplace=True,
            )
            self._df.reset_index(drop=True, inplace=True)

        if subset is not None:
            self._df = self._df.iloc[subset]
            self._df = self._df.reset_index(drop=True)

        self._structures = [
            os.path.join(self._structure_dir, f + ".cif") for f in self._df["info.qmof_id"]
        ]

        check_all_file_exists(self._structures)

        self._years = self._df["info.year"].values

        self._decorated_graph_hashes = self._df["info.decorated_graph_hash"].values
        self._undecorated_graph_hashes = self._df["info.undecorated_graph_hash"].values
        self._decorated_scaffold_hashes = self._df["info.decorated_scaffold_hash"].values
        self._undecorated_scaffold_hashes = self._df["info.undecorated_scaffold_hash"].values
        self._densities = self._df["info.density"].values
        self._labelnames = (c for c in self._df.columns if c.startswith("outputs."))
        self._featurenames = (c for c in self._df.columns if c.startswith("features."))
        self._infonames = (c for c in self._df.columns if c.startswith("info."))

    def get_subset(self, indices: Iterable[int]) -> "AbstractStructureDataset":
        """Get a subset of the dataset.

        Args:
            indices (Iterable[int]): indices of the structures to include.

        Returns:
            AbstractStructureDataset: a new dataset containing only the structures
                specified by the indices.
        """
        return QMOFDataset(
            version=self.version,
            drop_basename_duplicates=self._drop_basename_duplicates,
            drop_graph_duplicates=self._drop_graph_duplicates,
            subset=indices,
            flavor=self._flavor,
            drop_nan=self._drop_nan,
        )

    @property
    def available_info(self) -> Tuple[str]:
        return self._infonames

    @property
    def available_features(self) -> Tuple[str]:
        return self._featurenames

    @property
    def available_labels(self) -> Tuple[str]:
        return self._labelnames

    def get_labels(self, idx: Iterable[int], labelnames: Iterable[str] = None) -> np.ndarray:
        labelnames = labelnames if labelnames is not None else self._labelnames
        return self._df.iloc[idx][list(labelnames)].values

    @property
    def citations(self) -> Tuple[str]:
        return [
            "@article{Rosen2021,"
            "doi = {10.1016/j.matt.2021.02.015},"
            "url = {https://doi.org/10.1016/j.matt.2021.02.015},"
            "year = {2021},"
            "month = may,"
            "publisher = {Elsevier {BV}},"
            "volume = {4},"
            "number = {5},"
            "pages = {1578--1597},"
            "author = {Andrew S. Rosen and Shaelyn M. Iyer and Debmalya Ray "
            "and Zhenpeng Yao and Al{'{a}}n Aspuru-Guzik and Laura Gagliardi "
            "and Justin M. Notestein and Randall Q. Snurr},"
            "title = {Machine learning the quantum-chemical properties of"
            "metal{\textendash}organic frameworks for accelerated materials discovery},"
            "journal = {Matter}"
            "}",
            "@article{Rosen2022",
            "title={High-throughput predictions of metal--organic framework electronic properties:"
            " theoretical challenges, graph neural networks, and data exploration},"
            "author={Rosen, Andrew S and Fung, Victor and Huck, Patrick and O’Donnell, "
            "Cody T and Horton, Matthew K and Truhlar, Donald G and Persson, Kristin A "
            "and Notestein, Justin M and Snurr, Randall Q},"
            "journal={npj Computational Materials},"
            "volume={8},"
            "pages={112},"
            "year={2022},"
            "publisher={Nature Publishing Group}"
            "}",
        ]
