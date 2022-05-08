# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, Union

import numpy as np
from pymatgen.core import IStructure


class MOFStructureDataset(ABC):
    def __init__(
        self,
        labelnames: Tuple[str],
        version: str,
    ):
        pass

    def get_structure(self, idx: int) -> IStructure:
        return self._read_structure(self.files[idx])

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[IStructure, np.ndarray]:
        pass

    def _read_structure(self, filename) -> IStructure:
        return IStructure.from_file(filename)

    def _get_label(self, idx) -> np.ndarray:
        return self.labels[idx]

    def get_structures(self) -> Iterator:
        for file in self.files:
            yield self._read_structure(file)

    def get_labels(self) -> Iterator:
        for label in self.labels:
            yield label

    def get_year(self, idx: int) -> int:
        return self.years[idx]

    def get_label(self, idx: int) -> np.ndarray:
        return self._get_label(idx)

    def get_structures_and_labels(self, indices: List[int]) -> Iterator:
        for idx in indices:
            yield self.get_structure(idx), self.get_label(idx)

    @abstractmethod
    def get_time_based_split_indices(self, year: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_train_valid_test_split_indices(
        self, train_size: float, valid_size: float = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def available_labels(self) -> Tuple[str]:
        pass

    @property
    @abstractmethod
    def citations(self) -> List[str]:
        pass
