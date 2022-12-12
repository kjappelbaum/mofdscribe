# -*- coding: utf-8 -*-
"""Thermal Stability Dataset."""
import os
from typing import Collection, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mofdscribe.constants import MOFDSCRIBE_PYSTOW_MODULE
from mofdscribe.datasets.checks import check_all_file_exists, length_check
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.datasets.utils import compress_dataset

__all__ = ["ThermalStabilityDataset"]


class ThermalStabilityDataset(AbstractStructureDataset):
    """Thermal stability for a subset of CoRE MOFs.

    Reproduced from [Nandy2022]_.
    Nandy et al. (2022) digitized traces from thermogravimetric analysis.
    The decomposition temperature they determined in this way is reported in
    `outputs.assigned_T_decomp`.

    To reduce the risk of data leakage, we  (by default) also only keep one representative
    structure for a "base refcode" (i.e. the first five letters of a refcode).
    For instance, the base refcode for IGAHED001 is IGAHED. Structures with same
    base refcode but different refcodes are often different refinements, or measurements
    at different temperatures and hence chemically quite similar. For instance,
    the base refcode `UMODEH` would appear 21 times, `KEDJAG` 17 times, and `UMOYOM` 17 times
    in the CoRE dataset used by Moosavi et al.
    Additionally, we (by default) only keep one structure per "structure hash"
    which is an approximate graph-isomoprhism check, assuming the VESTA bond thresholds
    for the derivation of the structure graph (e.g. the structure
    graph of ULOMAL occurs 59 in the CoRE database used by Moosavi et al.).

    The years refer to the publication dates of the paper crossreferenced
    in the CSD entry of the structure.

    The available labels are:

        * `outputs.assigned_T_decomp`: Decomposition temperature in Kelvin.

    References::
        .. [Nandy2022] `Nandy, A.; Terrones, G.; Arunachalam, N.; Duan, C.;
            Kastner, D. W.; Kulik, H. J.
            MOFSimplify, Machine Learning Models with Extracted Stability Data
            of Three Thousand Metal–Organic Frameworks.
            Scientific Data 2022, 9 (1). <https://doi.org/10.1038/s41597-022-01181-0>`_
    """

    _files = {
        "v0.0.1": {
            "df": "https://zenodo.org/record/7428485/files/data.json?download=1",
            "structures": "https://zenodo.org/record/7428485/files/structures.tar.gz?download=1",
            "expected_length": 2039,
        }
    }

    def __init__(
        self,
        version: str = "v0.0.1",
        drop_basename_duplicates: bool = True,
        drop_graph_duplicates: bool = True,
        subset: Optional[Collection[int]] = None,
        drop_nan: bool = False,
    ):
        """Construct an instance of the ThermalStabilityDataset.

        Args:
            version (str): version number to use.
                Defaults to "v0.0.1".
            drop_basename_duplicates (bool): If True, keep only one structure
                per CSD basename. Defaults to True.
            drop_graph_duplicates (bool): If True, keep only one structure
                per decorated graph hash. Defaults to True.
            subset (Collection[int], optional): indices of the structures to include.
                Defaults to None.
            drop_nan (bool): If True, drop rows with NaN values in features or hashes.
                Defaults to True.

        Raises:
            ValueError: If the provided version number is not available.
        """
        self._drop_basename_duplicates = drop_basename_duplicates
        self._drop_nan = drop_nan
        self._drop_graph_duplicates = drop_graph_duplicates
        if version not in self._files:
            raise ValueError(
                f"Version {version} not available. Available versions: {list(self._files.keys())}"
            )
        self.version = version

        self._structure_dir = MOFDSCRIBE_PYSTOW_MODULE.ensure_untar(
            "thermal-stability",
            self.version,
            name="structures.tar.gz",
            url=self._files[version]["structures"],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                "thermal-stability", self.version, name="data.json", url=self._files[version]["df"]
            )
        ).reset_index(drop=True)

        compress_dataset(self._df)

        length_check(self._df, self._files[version]["expected_length"])

        if drop_basename_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["info.basename"])
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate basenames. New length {len(self._df)}"
            )
        if drop_graph_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=["info.decorated_graph_hash"])
            logger.debug(
                f"Dropped {old_len - len(self._df)} duplicate graphs. New length {len(self._df)}"
            )
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
            os.path.join(self._structure_dir, f + ".cif") for f in self._df["info.CoRE_name"]
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

    def get_subset(self, indices: Collection[int]) -> "AbstractStructureDataset":
        """Get a subset of the dataset.

        Args:
            indices (Collection[int]): indices of the structures to include.

        Returns:
            AbstractStructureDataset: a new dataset containing only the structures
                specified by the indices.
        """
        return ThermalStabilityDataset(
            version=self.version,
            drop_basename_duplicates=self._drop_basename_duplicates,
            drop_graph_duplicates=self._drop_graph_duplicates,
            subset=indices,
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

    def get_labels(self, idx: Collection[int], labelnames: Collection[str] = None) -> np.ndarray:
        labelnames = labelnames if labelnames is not None else self._labelnames
        return self._df.iloc[idx][list(labelnames)].values

    @property
    def citations(self) -> Tuple[str]:
        return [
            "@article{Chung2019,"
            "doi = {10.1021/acs.jced.9b00835},"
            "url = {https://doi.org/10.1021/acs.jced.9b00835},"
            "year = {2019},"
            "month = nov,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {64},"
            "number = {12},"
            "pages = {5985--5998},"
            "author = {Yongchul G. Chung and Emmanuel Haldoupis and Benjamin J. Bucior "
            "and Maciej Haranczyk and Seulchan Lee and Hongda Zhang and "
            "Konstantinos D. Vogiatzis and Marija Milisavljevic and Sanliang Ling "
            "and Jeffrey S. Camp and Ben Slater and J. Ilja Siepmann and "
            "David S. Sholl and Randall Q. Snurr},"
            "title = {Advances,  Updates,  and Analytics for the Computation-Ready, "
            "Experimental Metal{\textendash}Organic Framework Database: {CoRE} {MOF} 2019},"
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
            "author = {Yongchul G. Chung and Jeffrey Camp and "
            "Maciej Haranczyk and Benjamin J. Sikora and Wojciech Bury "
            "and Vaiva Krungleviciute and Taner Yildirim and Omar K. Farha "
            "and David S. Sholl and Randall Q. Snurr},"
            "title = {Computation-Ready,  Experimental Metal{\textendash}Organic Frameworks: "
            "A Tool To Enable High-Throughput Screening of Nanoporous Crystals},"
            "journal = {Chemistry of Materials}"
            "}",
            "@article{Nandy_2022,"
            "doi = {10.1038/s41597-022-01181-0},"
            "url = {https://doi.org/10.1038%2Fs41597-022-01181-0},"
            "year = 2022,"
            "month = {mar},"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {9},"
            "number = {1},"
            "author = {Aditya Nandy and Gianmarco Terrones and "
            "Naveen Arunachalam and Chenru Duan and David W. Kastner and Heather J. Kulik},"
            "title = {{MOFSimplify}, machine learning models with extracted stability data "
            "of three thousand metal{\textendash}organic frameworks},"
            "journal = {Sci Data}"
            "}",
        ]
