# -*- coding: utf-8 -*-
"""Structures from the ARABG database and labels from Moosavi et al."""
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mofdscribe.constants import MOFDSCRIBE_PYSTOW_MODULE
from mofdscribe.datasets.checks import check_all_file_exists, length_check
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.datasets.utils import compress_dataset

__all__ = ["ARABGDataset"]


class ARABGDataset(AbstractStructureDataset):
    """
    Exposes the ARABG dataset used in [Moosavi2020]_.

    The raw labels and structures can accessed also on
    `MaterialsCloud <https://archive.materialscloud.org/record/2020.67>`_.

    It is a subset of the ARABG database [Anderson2018]_ with labels
    computed by Moosavi et al.
    Those labels deviate in value and computational
    approach from the original labels in [Boyd2019]_ but are consistent
    with the labels for the other databases in [Moosavi2020]_.

    The available labels are:
        * 'pure_CO2_kH': Henry coefficient of CO2 obtained by Widom method in mol kg-1 Pa-1
        * 'pure_CO2_widomHOA': Heat of adsorption of CO2 obtained by Widom method in
        * 'pure_methane_kH': Henry coefficient of methane obtained by Widom method in mol kg-1 Pa-1
        * 'pure_methane_widomHOA': Heat of adsorption of methane obtained by Widom method
        * 'pure_uptake_CO2_298.00_15000': Pure CO2 uptake at 298.00 K and 15000 Pa in mol kg-1
        * 'pure_uptake_CO2_298.00_1600000': Pure CO2 uptake at 298.00 K and 1600000 Pa in mol kg-1
        * 'pure_uptake_methane_298.00_580000': Pure methane uptake at 298.00 K and 580000 Pa in mol kg-1
        * 'pure_uptake_methane_298.00_6500000': Pure methane uptake at 298.00 K and 6500000 Pa in mol kg-1
        * 'logKH_CO2': Logarithm of Henry coefficient of CO2 obtained by Widom method
        * 'logKH_CH4': Logarithm of Henry coefficient of methane obtained by Widom method
        * 'CH4DC': CH4 deliverable capacity in vSTP/v
        * 'CH4HPSTP': CH4 high pressure uptake in standard temperature and pressure in vSTP/v
        * 'CH4LPSTP': CH4 low pressure uptake in standard temperature and pressure in vSTP/v

    .. note::

        The ARABG structures are hypothetical MOFs, therefore the following caveats apply:

        * It is well known that the data distribution can be quite different from experimental
          structures
        * The structures were only optimized using the UFF force field [UFF]_
        * A time-based split cannot be used for hypothetical structures


    .. admonition:: Information about building blocks
        :class: hint

        This dataset exposed information about the building blocks of the MOFs.
        You might find this useful for grouped-cross-validation (as MOFs with same building-blocks
        and/or net are not really independent).

        You find this info under also in the `info.rcsr_code`, `info.metal_bb`, and
        `info.organic_bb`, `info.functional_group` columns.

    .. warning:: Danger of data leakage

        Cross validation for MOFs with same building-blocks and/or net is notoriously
        difficult. Since all combinations of building-blocks and/or net are considered,
        it is not trivial to find completely independent groups.

    References:
        .. [Moosavi2020] `Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet, J. P.; Boyd, P. G.; Lee,
            Y.; Smit, B.; Kulik, H. J. Understanding the Diversity of the Metal-Organic Framework Ecosystem.
            Nature Communications 2020, 11 (1), 4068. <https://doi.org/10.1038/s41467-020-17755-8>`_

        .. [Anderson2018] `Anderson, R.; Rodgers, J.; Argueta, E.; Biong, A.; Gómez-Gualdrón, D. A.
            Role of Pore Chemistry and Topology in the CO2 Capture Capabilities of MOFs:
            From Molecular Simulation to Machine Learning. Chemistry of Materials 2018, 30 (18), 6325–6337.
            <https://doi.org/10.1021/acs.chemmater.8b02257>_
    """

    _files = {
        "v0.0.1": {
            "df": "https://zenodo.org/record/7032106/files/data.json?download=1",
            "structures": "https://zenodo.org/record/7032106/files/structures.tar.gz?download=1",
            "expected_length": 370,
        }
    }

    def __init__(
        self,
        version: str = "v0.0.1",
        drop_graph_duplicates: bool = True,
        subset: Optional[Iterable[int]] = None,
    ):
        """Construct an instance of the CoRE dataset.

        Args:
            version (str): version number to use.
                Defaults to "v0.0.1".
            drop_graph_duplicates (bool): If True, keep only one structure
                per decorated graph hash. Defaults to True.
            subset (Iterable[int], optional): indices of the structures to include.
                Defaults to None.

        Raises:
            ValueError: If the provided version number is not available.
        """
        self._drop_graph_duplicates = drop_graph_duplicates
        if version not in self._files:
            raise ValueError(
                f"Version {version} not available. Available versions: {list(self._files.keys())}"
            )
        self.version = version

        self._structure_dir = MOFDSCRIBE_PYSTOW_MODULE.ensure_untar(
            "ARABG",
            self.version,
            name="structures.tar.gz",
            url=self._files[version]["structures"],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                "ARABG", self.version, name="data.json", url=self._files[version]["df"]
            )
        ).reset_index(drop=True)
        compress_dataset(self._df)

        length_check(self._df, self._files[version]["expected_length"])

        if drop_graph_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["info.decorated_graph_hash"])
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate graphs. New length {len(self._df)}"
            )
        self._df = self._df.reset_index(drop=True)
        if subset is not None:
            self._df = self._df.iloc[subset]
            self._df = self._df.reset_index(drop=True)

        self._structures = [
            os.path.join(self._structure_dir, f + ".cif") for f in self._df["info.name"]
        ]

        check_all_file_exists(self._structures)

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
        return ARABGDataset(
            version=self.version,
            drop_graph_duplicates=self._drop_graph_duplicates,
            subset=indices,
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
            "@article{Moosavi2020,"
            "doi = {10.1038/s41467-020-17755-8},"
            "url = {https://doi.org/10.1038/s41467-020-17755-8},"
            "year = {2020},"
            "month = aug,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {11},"
            "number = {1},"
            "author = {Seyed Mohamad Moosavi and Aditya Nandy and Kevin Maik Jablonka "
            "and Daniele Ongari and Jon Paul Janet and Peter G. Boyd and Yongjin Lee "
            "and Berend Smit and Heather J. Kulik},"
            "title = {Understanding the diversity of the metal-organic framework ecosystem},"
            "journal = {Nature Communications}"
            "}",
            "@article{Anderson_2018,"
            "doi = {10.1021/acs.chemmater.8b02257},"
            "url = {https://doi.org/10.1021%2Facs.chemmater.8b02257},"
            "year = 2018,"
            "month = {aug},"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {30},"
            "number = {18},"
            "pages = {6325--6337},"
            "author = {Ryther Anderson and Jacob Rodgers and Edwin Argueta "
            "and Achay Biong and Diego A. G{'{o}}mez-Gualdr{'{o}}n},"
            "title = {Role of Pore Chemistry and Topology in the {CO}$_2$ "
            "Capture Capabilities of {MOFs}: From Molecular Simulation to Machine Learning},"
            "journal = {Chem. Mater.}"
            "}",
        ]
