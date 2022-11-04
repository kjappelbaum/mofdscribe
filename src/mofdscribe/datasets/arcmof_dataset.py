# -*- coding: utf-8 -*-
"""Structures from the ARC-MOF dataset."""
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mofdscribe.constants import MOFDSCRIBE_PYSTOW_MODULE
from mofdscribe.datasets.checks import check_all_file_exists, length_check
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.datasets.utils import compress_dataset

__all__ = ["ARCMOFDataset"]


class ARCMOFDataset(AbstractStructureDataset):
    """
    Implements access to a subset of structures and labels of the ARC-MOF dataset [Burner2022]_.

    The subset consistes of the structures for which the authors reported
    process properties and for which we could compute graph hashes
    and features.

    .. warning::

        ARC-MOF is  "a database of ~280,000 MOFs which have been either
        experimentally characterized or computationally generated,
        spanning all publicly available MOF databases" [Burner2022]_.

        Therefore, there will be significant overlap with the other datasets.

    References:
        .. [Burner2022] `Burner, J.; Luo, J.; White, A.;
            Mirmiran, A.; Kwon, O.; Boyd, P. G.;
            Maley, S.; Gibaldi, M.; Simrod, S.;
            Ogden, V.; Woo, T. K.
            ChemRxiv 2022 <https://chemrxiv.org/engage/chemrxiv/article-details/62e04636cf661270d7b615c1>`_
    """

    _files = {
        "v0.0.1": {
            "df": "https://zenodo.org/record/7032350/files/data.json?download=1",
            "structures": "https://zenodo.org/record/7032350/files/structures.tar.gz?download=1",
            "expected_length": 22452,
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
            "ARCMOF",
            self.version,
            name="structures.tar.gz",
            url=self._files[version]["structures"],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                "ARCMOF", self.version, name="data.json", url=self._files[version]["df"]
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
        return ARCMOFDataset(
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
            "@article{Burner_2022,"
            "doi = {10.26434/chemrxiv-2022-mvr06},"
            "url = {https://doi.org/10.26434%2Fchemrxiv-2022-mvr06},"
            "year = 2022,"
            "month = {aug},"
            "publisher = {American Chemical Society ({ACS})},"
            "author = {Jake Burner and Jun Luo and Andrew White "
            "and Adam Mirmiran and Ohmin Kwon and Peter G. Boyd "
            "and Stephen Maley and Marco Gibaldi and Scott Simrod "
            "and Victoria Ogden and Tom K. Woo},"
            "title = {{ARC}-{MOF}: A Diverse Database of "
            "Metal-Organic Frameworks with {DFT}-Derived Partial "
            "Atomic Charges and Descriptors for Machine Learning}}"
        ]
