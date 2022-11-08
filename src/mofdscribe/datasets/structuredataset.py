# -*- coding: utf-8 -*-
"""Interface for creating a custom StructureDataset."""
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from mofdscribe.datasets.checks import check_all_file_exists
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.datasets.utils import compress_dataset
from mofdscribe.types import PathType
from loguru import logger

__all__ = ["StructureDataset", "FrameDataset"]


class StructureDataset(AbstractStructureDataset):
    """Custom dataset class for loading structures from a files"""

    def __init__(
        self,
        files: Iterable[PathType],
        df: Optional[pd.DataFrame] = None,
        structure_name_column: Optional[str] = None,
        year_column: Optional[str] = None,
        label_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        decorated_graph_hash_column: Optional[str] = None,
        undecorated_graph_hash_column: Optional[str] = None,
        decorated_scaffold_hash_column: Optional[str] = None,
        undecorated_scaffold_hash_column: Optional[str] = None,
        density_column: Optional[str] = None,
    ):
        """Initialize a dataset.

        Args:
            files (Iterable[PathType]): List of files to load structures from.
            df (Optional[pd.DataFrame], optional): Dataframe containing the structures.
                Defaults to None.
            structure_name_column (str): Name of the column containing the structure names.
                Defaults to None.
            year_column (str, optional): Name of the column containing the year of the structure.
                Defaults to None.
            label_columns (Optional[List[str]], optional): List of columns containing the labels.
                Defaults to None.
            feature_columns (Optional[List[str]], optional): List of columns containing the features.
                Defaults to None.
            decorated_graph_hash_column (str, optional): Name of the column containing the decorated graph hash.
                Defaults to None.
            undecorated_graph_hash_column (str, optional): Name of the column containing the undecorated graph hash.
                Defaults to None.
            decorated_scaffold_hash_column (str, optional): Name of the column containing the decorated scaffold hash.
                Defaults to None.
            undecorated_scaffold_hash_column (str, optional): Name of the column containing
                the undecorated scaffold hash.
                Defaults to None.
            density_column (str, optional): Name of the column containing the density of the structure.
                Defaults to None.
        """
        super().__init__()
        self._df = df
        if self._df is not None:
            compress_dataset(self._df)
            self._structures = [
                f for f in files if Path(f).stem in self._df[structure_name_column].values
            ]
        else:
            self._structures = files
        self._structure_name_column = structure_name_column
        check_all_file_exists(self._structures)

        self._year_column = year_column
        self._label_columns = list(label_columns) if label_columns is not None else tuple()
        self._feature_columns = list(feature_columns) if feature_columns is not None else tuple()
        self._decorated_graph_hash_column = decorated_graph_hash_column
        self._undecorated_graph_hash_column = undecorated_graph_hash_column
        self._decorated_scaffold_hash_column = decorated_scaffold_hash_column
        self._undecorated_scaffold_hash_column = undecorated_scaffold_hash_column
        self._density_column = density_column

        self._years = None if year_column is None else self._df[year_column].values
        self._labels = None if label_columns is None else self._df[label_columns].values
        self._decorated_graph_hashes = (
            None
            if decorated_graph_hash_column is None
            else self._df[decorated_graph_hash_column].values
        )
        self._undecorated_graph_hashes = (
            None
            if undecorated_graph_hash_column is None
            else self._df[undecorated_graph_hash_column].values
        )
        self._decorated_scaffold_hashes = (
            None
            if decorated_scaffold_hash_column is None
            else self._df[decorated_scaffold_hash_column].values
        )
        self._undecorated_scaffold_hashes = (
            None
            if undecorated_scaffold_hash_column is None
            else self._df[undecorated_scaffold_hash_column].values
        )
        self._densities = None if density_column is None else self._df[density_column].values

    def __len__(self):
        """Return number of structures in the dataset."""
        return len(self._structures)

    @property
    def available_features(self) -> List[str]:
        return self._featurenames

    @property
    def available_labels(self) -> List[str]:
        return self._labelnames

    def get_labels(self, idx: Iterable[int], labelnames: Iterable[str] = None) -> np.ndarray:
        labelnames = labelnames if labelnames is not None else self._labelnames
        return self._df.iloc[idx][list(labelnames)].values

    @classmethod
    def from_folder_and_dataframe(
        cls,
        folder: PathType,
        extension: str = "cif",
        dataframe: Optional[pd.DataFrame] = None,
        structure_name_column: Optional[str] = None,
        year_column: Optional[str] = None,
        label_columns: Optional[List[str]] = None,
        decorated_graph_hash_column: Optional[str] = None,
        undecorated_graph_hash_column: Optional[str] = None,
        decorated_scaffold_hash_column: Optional[str] = None,
        undecorated_scaffold_hash_column: Optional[str] = None,
        density_column: Optional[str] = None,
    ) -> "StructureDataset":
        """Create a dataset from a folder and a dataframe.

        Args:
            folder (PathType): Path to the folder containing the structures.
            extension (str): Extension of the files. Defaults to 'cif'.
            dataframe (Optional[pd.DataFrame], optional): Dataframe containing the structures.
                Defaults to None.
            structure_name_column (str): Name of the column containing the structure names.
                Defaults to None.
            year_column (str, optional): Name of the column containing the year of the structure.
                Defaults to None.
            label_columns (Optional[List[str]], optional): List of columns containing the labels.
                Defaults to None.
            decorated_graph_hash_column (str, optional): Name of the column containing the decorated graph hash.
                Defaults to None.
            undecorated_graph_hash_column (str, optional): Name of the column containing the undecorated graph hash.
                Defaults to None.
            decorated_scaffold_hash_column (str, optional): Name of the column containing the decorated scaffold hash.
                Defaults to None.
            undecorated_scaffold_hash_column (str, optional): Name of the column containing the undecorated scaffold
                hash. Defaults to None.
            density_column (str, optional): Name of the column containing the density of the structure.
                Defaults to None.

        Returns:
            StructureDataset: Dataset containing the structures.
        """
        all_files = list(Path(folder).rglob(f"*.{extension}"))
        return cls(
            all_files,
            dataframe,
            structure_name_column,
            year_column,
            label_columns,
            decorated_graph_hash_column,
            undecorated_graph_hash_column,
            decorated_scaffold_hash_column,
            undecorated_scaffold_hash_column,
            density_column,
        )


class FrameDataset(AbstractStructureDataset):
    """Dataset containing structure information read from a dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        structure_name_column: str,
        year_column: Optional[str] = None,
        label_columns: Optional[List[str]] = None,
        decorated_graph_hash_column: Optional[str] = None,
        undecorated_graph_hash_column: Optional[str] = None,
        decorated_scaffold_hash_column: Optional[str] = None,
        undecorated_scaffold_hash_column: Optional[str] = None,
        density_column: Optional[str] = None,
    ):
        """Initialize the dataset.

        Args:
            df (pd.DataFrame): Dataframe containing the structures.
            structure_name_column (str): Name of the column containing the structure names.
            year_column (str, optional): Name of the column containing the year of the structure.
                Defaults to None.
            label_columns (Optional[List[str]], optional): List of columns containing the labels.
                Defaults to None.
            decorated_graph_hash_column (str, optional): Name of the column containing the decorated graph hash.
                Defaults to None.
            undecorated_graph_hash_column (str, optional): Name of the column containing the undecorated graph hash.
                Defaults to None.
            decorated_scaffold_hash_column (str, optional): Name of the column containing the decorated scaffold hash.
                Defaults to None.
            undecorated_scaffold_hash_column (str, optional): Name of the column containing the undecorated scaffold
                hash. Defaults to None.
            density_column (str, optional): Name of the column containing the density of the structure.
                Defaults to None.
        """
        super().__init__()
        logger.warning("FrameDataset support is experimental. Some splitter integrations may not work.")
        self._df = df
        compress_dataset(self._df)
        self._structure_name_column = structure_name_column
        self._year_column = year_column
        self._label_columns = list(label_columns) if label_columns is not None else tuple()
        self._decorated_graph_hash_column = decorated_graph_hash_column
        self._undecorated_graph_hash_column = undecorated_graph_hash_column
        self._decorated_scaffold_hash_column = decorated_scaffold_hash_column
        self._undecorated_scaffold_hash_column = undecorated_scaffold_hash_column
        self._density_column = density_column

        self._years = None if year_column is None else self._df[year_column]
        self._labels = None if label_columns is None else self._df[label_columns].values
        self._decorated_graph_hashes = (
            None
            if decorated_graph_hash_column is None
            else self._df[decorated_graph_hash_column].values
        )
        self._undecorated_graph_hashes = (
            None
            if undecorated_graph_hash_column is None
            else self._df[undecorated_graph_hash_column].values
        )
        self._decorated_scaffold_hashes = (
            None
            if decorated_scaffold_hash_column is None
            else self._df[decorated_scaffold_hash_column].values
        )
        self._undecorated_scaffold_hashes = (
            None
            if undecorated_scaffold_hash_column is None
            else self._df[undecorated_scaffold_hash_column].values
        )
        self._densities = None if density_column is None else self._df[density_column].values

    def __len__(self):
        """Return number of structures in the dataset."""
        return len(self._df)

    @property
    def available_features(self) -> List[str]:
        return self._featurenames

    @property
    def available_labels(self) -> List[str]:
        return self._labelnames

    def get_labels(self, idx: Iterable[int], labelnames: Iterable[str] = None) -> np.ndarray:
        labelnames = labelnames if labelnames is not None else self._labelnames
        return self._df.iloc[idx][list(labelnames)].values