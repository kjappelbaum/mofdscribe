# -*- coding: utf-8 -*-
"""Interface for creating a custom StructureDataset."""
from mofdscribe.datasets.dataset import AbstractStructureDataset
from mofdscribe.types import PathType
from typing import Iterable, List
import pandas as pd 
from mofdscribe.datasets.utils import compress_dataset

__all__ = ["StructureDataset"]

class StructureDataset(AbstractStructureDataset):
    """Custom dataset class for loading structures from a files"""
    def __init__(self, files: Iterable[PathType], df: pd.DataFrame, year_column: str, label_columns: List[str], decorated_graph_hash_column: str, undecorated_graph_hash_column: str, decorated_scaffold_hash_column: str, undecorated_scaffold_hash_column: str, density_column: str):
        super().__init__()
        self._files = files
        self._df = df
        compress_dataset(self._df)
        self._year_column = year_column
        self._label_columns = label_columns

    @classmethod
    def from_folder_and_dataframe(cls, folder, dataframe, extension, year_column, label_columns):
        ...