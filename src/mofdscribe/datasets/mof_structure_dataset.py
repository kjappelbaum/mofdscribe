# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, Union

import numpy as np
from pymatgen.core import IStructure

from .checks import check_all_file_exists, length_check


class MOFStructureDataset(ABC):
    def __init__(
        self,
        labelnames: Tuple[str],
        version: str,
    ):
        pass

    def _precheck(self) -> None:
        length_check(self._df, self._expected_length)
        check_all_file_exists(self._files)

    def get_structure(self, idx: int) -> IStructure:
        return self._read_structure(self._files[idx])

    def __getitem__(self, idx: int) -> Tuple[IStructure, np.ndarray]:
        return (self.get_structure(idx), self.get_label(idx))

    def _read_structure(self, filename) -> IStructure:
        return IStructure.from_file(filename)

    def _get_label(self, idx) -> np.ndarray:
        return self._labels.iloc[idx].values

    def get_structures(self) -> Iterator:
        for file in self._files:
            yield self._read_structure(file)

    def get_labels(self) -> Iterator:
        for label in self._labels:
            yield label

    def get_year(self, idx: int) -> int:
        return self._years[idx]

    def get_label(self, idx: int) -> np.ndarray:
        return self._get_label(idx)

    def get_structures_and_labels(self, indices: List[int]) -> Iterator:
        for idx in indices:
            yield self.get_structure(idx), self.get_label(idx)

    def get_time_based_split_indices(
        self, year: int, year_valid: Union[int, None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if year_valid is None:
            if year_valid >= year:
                raise ValueError("Year valid must be smaller than year.")

        train = np.where(self._df["year"] <= year)
        test = np.where(self._df["year"] > year)

        if year_valid is not None:
            valid = np.where((self._df["year"] < year) & (self._df["year"] > year_valid))
            return train, valid, test

        return train, test

    @abstractmethod
    def __len__(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def available_labels(self) -> Tuple[str]:
        pass

    @property
    @abstractmethod
    def citations(self) -> List[str]:
        pass
