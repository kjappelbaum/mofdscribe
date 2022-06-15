# -*- coding: utf-8 -*-
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .checks import check_all_file_exists, length_check
from .dataset import StructureDataset
from ..constants import MOFDSCRIBE_PYSTOW_MODULE


class CoREDataset(StructureDataset):
    _files = {
        "v0.0.1": {
            "df": "https://www.dropbox.com/s/i2khfte4s6mcg30/data.json?dl=1",
            "structures": "https://www.dropbox.com/s/1v7zknesttixw68/structures.tar.gz?dl=1",
            "expected_length": 8821,
        }
    }

    def __init__(
        self,
        version: str = "v0.0.1",
        drop_basename_duplicates: bool = True,
        drop_graph_duplicates: bool = True,
    ):

        if not version in self._files:
            raise ValueError(
                f"Version {version} not available. Available versions: {list(self._files.keys())}"
            )
        self.version = version

        self._structure_dir = MOFDSCRIBE_PYSTOW_MODULE.ensure_untar(
            "CoRE",
            self.version,
            name="structures.tar.gz",
            url=self._files[version]["structures"],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                "CoRE", self.version, name="data.json", url=self._files[version]["df"]
            )
        ).reset_index(drop=True)

        length_check(self._df, self._files[version]["expected_length"])

        if drop_basename_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["basename"])
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate basenames. New length {len(self._df)}"
            )
        if drop_graph_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["hash"])
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate graphs. New length {len(self._df)}"
            )
        self._df = self._df.reset_index(drop=True)
        self._structures = [
            os.path.join(self._structure_dir, f + ".cif") for f in self._df["name_y"]
        ]

        check_all_file_exists(self._structures)

        self._years = self._df["year"]
        self._decorated_graph_hashes = self._df["hash"]
        self._undecorated_graph_hashes = self._df["undecorated_hash"]
        self._decorated_scaffold_hashes = self._df["scaffold_hash"]
        self._undecorated_scaffold_hashes = self._df["undecorated_scaffold_hash"]
        self._densities = self._df["density_x"]
        self._labelnames = tuple(
            [
                "pure_CO2_kH",
                "pure_CO2_widomHOA",
                "pure_methane_kH",
                "pure_methane_widomHOA",
                "pure_uptake_CO2_298.00_15000",
                "pure_uptake_CO2_298.00_1600000",
                "pure_uptake_methane_298.00_580000",
                "pure_uptake_methane_298.00_6500000",
                "logKH_CO2",
                "logKH_CH4",
                "CH4DC",
                "CH4HPSTP",
                "CH4LPSTP",
            ]
        )

    @property
    def available_labels(self) -> Tuple[str]:
        return self._labelnames

    def get_labels(
        self, idx: Iterable[int], labelnames: Optional[Iterable[str]] = None
    ) -> np.ndarray:
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
            "author = {Seyed Mohamad Moosavi and Aditya Nandy and Kevin Maik Jablonka and Daniele Ongari and Jon Paul Janet and Peter G. Boyd and Yongjin Lee and Berend Smit and Heather J. Kulik},"
            "title = {Understanding the diversity of the metal-organic framework ecosystem},"
            "journal = {Nature Communications}"
            "}",
            "@article{Chung2019,"
            "doi = {10.1021/acs.jced.9b00835},"
            "url = {https://doi.org/10.1021/acs.jced.9b00835},"
            "year = {2019},"
            "month = nov,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {64},"
            "number = {12},"
            "pages = {5985--5998},"
            "author = {Yongchul G. Chung and Emmanuel Haldoupis and Benjamin J. Bucior and Maciej Haranczyk and Seulchan Lee and Hongda Zhang and Konstantinos D. Vogiatzis and Marija Milisavljevic and Sanliang Ling and Jeffrey S. Camp and Ben Slater and J. Ilja Siepmann and David S. Sholl and Randall Q. Snurr},"
            "title = {Advances,  Updates,  and Analytics for the Computation-Ready,  Experimental Metal{\textendash}Organic Framework Database: {CoRE} {MOF} 2019},"
            r"journal = {Journal of Chemical {\&}amp$\mathsemicolon$ Engineering Data}"
            "}",
            "@article{Chung2014,"
            "doi = {10.1021/cm502594j},"
            "url = {https://doi.org/10.1021/cm502594j},"
            "year = {2014},"
            "month = oct,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {26},"
            "number = {21},"
            "pages = {6185--6192},"
            "author = {Yongchul G. Chung and Jeffrey Camp and Maciej Haranczyk and Benjamin J. Sikora and Wojciech Bury and Vaiva Krungleviciute and Taner Yildirim and Omar K. Farha and David S. Sholl and Randall Q. Snurr},"
            "title = {Computation-Ready,  Experimental Metal{\textendash}Organic Frameworks: A Tool To Enable High-Throughput Screening of Nanoporous Crystals},"
            "journal = {Chemistry of Materials}"
            "}",
        ]
