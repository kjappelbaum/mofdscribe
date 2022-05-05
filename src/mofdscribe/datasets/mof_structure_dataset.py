from abc import ABC, abstractmethod
from typing import Iterator, Union, List, Tuple
from pymatgen.core import IStructure
import numpy as np


class MOFStructureDataset(ABC):
    def __init__(self, version: str):
        pass

    def get_structure(self, idx: int, name: str) -> IStructure:
        pass

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
        pass

    def get_label(self, idx: int) -> np.ndarray:
        return self._get_label(idx)

    def get_time_based_split(self, year: int) -> Tuple[List[IStructure], np.ndarray]:
        pass

    def get_train_valid_test_split(
        self, train_size: float, valid_size: float = 0
    ) -> Tuple[
        Tuple[List[IStructure], np.ndarray],
        Tuple[List[IStructure], np.ndarray],
        Tuple[List[IStructure], np.ndarray],
    ]:
        pass

    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def citations(self) -> List[str]:
        pass
