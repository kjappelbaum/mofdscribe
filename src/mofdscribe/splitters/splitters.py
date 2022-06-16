# -*- coding: utf-8 -*-
"""Classes that help to perform cross-validation."""
from collections import Counter
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..datasets.dataset import StructureDataset

__all__ = (
    "DensitySplitter",
    "HashSplitter",
    "TimeSplitter",
    "Splitter",
    "RandomSplitter",
    "FingerprintSplitter",
)


class Splitter:
    """Base class for splitters."""

    def train_valid_test_split(
        self,
        ds: StructureDataset,
        frac_train: float,
        frac_valid: float,
        sample_frac: Optional[float] = 1.0,
        shuffle: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
        """
        Get train, valid, and test indices.

        Args:
            ds (StructureDataset): a mofdscribe dataset
            frac_train (float): fraction of the data to use for training
            frac_valid (float): fraction of the data to use for validation.
                1 - frac_train - frac_valid = fraction of the data to use for
                testing
            sample_frac (float, optional): Fraction by which the full dataset is
                downsampled (randomly). Can be useful for debugging. Defaults to 1.0.
            shuffle (bool, optional): If True, then the splitters attempt to
            shuffle the data at every possible step.
                For the splitters that internally perform a sorting operation,
                this means that the ties will be shuffled. In any case, if
                shuffle is set to True, then the folds will be shuffled after
                the split. Defaults to True.
            **kwargs: additional arguments for the splitter

        Raises:
            ValueError: if frac_train + frac_valid > 1.0 ValueError: if
            sample_frac > 1.0

        Returns:
            Tuple[Iterable[int], Iterable[int], Iterable[int]]: train, valid,
            test indices
        """
        for frac in [frac_train, frac_valid]:
            if frac >= 1.0:
                raise ValueError("frac_train and frac_valid must be < 1.0")
        if sample_frac > 1.0:
            raise ValueError("sample_frac must be <= 1.0")
        number_of_points = int(len(ds) * sample_frac)
        number_of_train_points = int(frac_train * number_of_points)
        number_of_valid_points = int(frac_valid * number_of_points)

        indices = self.get_sorted_indices(ds, shuffle=shuffle)

        train_inds = indices[:number_of_train_points]
        valid_inds = indices[
            number_of_train_points : number_of_train_points + number_of_valid_points
        ]
        test_inds = indices[number_of_train_points + number_of_valid_points :]

        if shuffle:
            np.random.shuffle(train_inds)
            np.random.shuffle(valid_inds)
            np.random.shuffle(test_inds)
        return train_inds, valid_inds, test_inds

    def train_test_split(
        self,
        ds: StructureDataset,
        frac_train: float,
        sample_frac: Optional[float] = 1.0,
        shuffle: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[Iterable[int], Iterable[int]]:
        """

        Args:
            ds (StructureDataset): mofdscribe dataset
            frac_train (float): fraction of the data to use for training
            sample_frac (float, optional): Fraction by which the full dataset
                is downsampled (randomly).
                Can be useful for debugging. Defaults to 1.0.
            shuffle (bool, optional): If True, then the splitters attempt to
            shuffle the data at every possible step.
                For the splitters that internally perform a sorting operation,
                this means that the ties will be shuffled. In any case, if
                shuffle is set to True, then the folds will be shuffled after
                the split. Defaults to True.
            **kwargs: Aditional options for the splitter

        Raises:
            ValueError: if frac_train + frac_valid > 1.0 ValueError: if
            sample_frac > 1.0

        Returns:
            Tuple[Iterable[int], Iterable[int]]: train, test indices
        """
        if frac_train >= 1.0:
            raise ValueError("frac_train and frac_test must be < 1.0")

        if sample_frac > 1.0:
            raise ValueError("sample_frac must be <= 1.0")
        number_of_points = int(len(ds) * sample_frac)
        number_of_train_points = int(frac_train * number_of_points)

        indices = self.get_sorted_indices(ds, shuffle=shuffle)

        train_inds = indices[:number_of_train_points]

        test_inds = indices[number_of_train_points:]

        if shuffle:
            np.random.shuffle(train_inds)
            np.random.shuffle(test_inds)

        return train_inds, test_inds

    def k_fold(self, ds: StructureDataset, k: int, shuffle: Optional[bool] = True):
        """Split the data into k folds."""
        indices = self.get_sorted_indices(ds, shuffle=shuffle)
        indices = np.array(indices)
        fold_sizes = np.full(k, len(ds) // k, dtype=int)
        fold_sizes[: len(ds) % k] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_index = indices[start:stop]

            test_mask = np.zeros(len(ds), dtype=bool)
            test_mask[test_index] = True
            train_fold = indices[np.logical_not(test_mask)]
            test_fold = indices[test_mask]
            if shuffle:
                np.random.shuffle(train_fold)
                np.random.shuffle(test_fold)

            yield train_fold, test_fold
            current = stop


class HashSplitter(Splitter):
    """Splitter that uses graph hashes to split the data in more stringent ways.

    Note that the hashes we use do not allow for a meaningful measure of
    similarity.

    However, we can sort the data by the hash and make sure that the same hash
    is only occuring in one set. Moreover, we take care that the largest groups
    of duplicated hashes are in the training set.
    """

    def __init__(
        self,
        hash_type: Optional[str] = "undecorated_scaffold_hash",
    ) -> None:
        """Initialize a HashSplitter.

        Args:
            hash_type (str, optional): Hash type to use. Must be one of the
                following:
                * undecorated_scaffold_hash
                * decorated_graph_hash
                * decorated_scaffold_hash
                * undecorated_graph_hash
                Defaults to "undecorated_scaffold_hash".
        """
        self.hash_type = hash_type
        super().__init__()

    def get_hashes(self, ds: StructureDataset) -> Iterable[str]:
        """Retrieve the list of hashes from the dataset

        Args:
            ds (StructureDataset): mofdscribe dataset.

        Raises:
            ValueError: If the hash type is not one of the following:
            * undecorated_scaffold_hash
            * decorated_graph_hash
            * decorated_scaffold_hash
            * undecorated_graph_hash

        Returns:
            Iterable[str]: list of hashes
        """
        number_of_points = len(ds)
        if self.hash_type == "undecorated_scaffold_hash":
            hashes = ds.get_undecorated_scaffold_hashes(range(number_of_points))
        elif self.hash_type == "decorated_graph_hash":
            hashes = ds.get_decorated_graph_hashes(range(number_of_points))
        elif self.hash_type == "decorated_scaffold_hash":
            hashes = ds.get_decorated_scaffold_hashes(range(number_of_points))
        elif self.hash_type == "undecorated_graph_hash":
            hashes = ds.get_undecorated_graph_hashes(range(number_of_points))
        else:
            raise ValueError(f"Unknown hash type: {self.hash_type}")

        return hashes

    def get_sorted_indices(
        self, ds: StructureDataset, shuffle: Optional[bool] = True
    ) -> Iterable[int]:
        """Create a sorted list of indices based on the hashes of the structures.

        Args:
            ds (StructureDataset): mofdscribe dataset
            shuffle (bool, optional): If true, shuffle ties (identical hash).
                Defaults to True.

        Returns:
            Iterable[int]: sorted indices
        """
        hashes = self.get_hashes(ds)
        # we use these numbers to shuffle the data in case of ties
        random_numbers = np.arange(len(hashes))
        if shuffle:
            np.random.shuffle(random_numbers)

        c = Counter(hashes)
        t = [(c[v], v, i, random_numbers[i]) for i, v in enumerate(hashes)]
        t.sort(reverse=True, key=lambda x: (x[0], x[3]))
        indices = [i for _, _, i, _ in t]

        return indices


class DensitySplitter(Splitter):
    """Splitter that uses the density of the structures to split the data.

    For this, we sort structures according to their density and then split the
    data. This ensures that the validation is quite stringent as the different
    folds will have different densities.
    """

    def __init__(
        self,
        ascending: Optional[bool] = True,
    ) -> None:
        """Initialize the DensitySplitter class.

        Args:
            ascending (bool, optional): If True, sort densities ascending.
                Defaults to True.
        """
        self.ascending = ascending
        super().__init__()

    def get_sorted_indices(
        self, ds: StructureDataset, shuffle: Optional[bool] = True
    ) -> Iterable[int]:
        """Create a sorted list of indices based on the density of the structures.

        Args:
            ds (StructureDataset): mofdscribe dataset
            shuffle (bool, optional): If true, shuffle ties (identical densities).
                Defaults to True.

        Returns:
            Iterable[int]: sorted indices
        """
        densities = ds.get_densities(range(len(ds)))
        # we use these numbers to shuffle the data in case of ties
        random_numbers = np.arange(len(densities))
        if shuffle:
            np.random.shuffle(random_numbers)

        t = [(v, i, random_numbers[i]) for i, v in enumerate(densities)]
        t.sort(reverse=not self.ascending, key=lambda x: (x[0], x[2]))
        indices = [i for _, i, _ in t]

        return indices


class TimeSplitter(Splitter):
    """This splitter sorts structures according to their publication date.

    That is, the training set will contain structures that are "older" (have
    been discovered earlier) than the ones in the test set.
    """

    def __init__(
        self,
        ascending: Optional[bool] = True,
    ) -> None:
        """Initialize the TimeSplitter class

        Args:
            ascending (bool, optional): If True, sort times ascending.
                Defaults to True.
        """
        self.ascending = ascending
        super().__init__()

    def get_sorted_indices(self, ds, shuffle: Optional[bool] = True):
        """Create a sorted list of indices based on the structures' year of publication.

        Args:
            ds (StructureDataset): mofdscribe dataset
            shuffle (bool, optional): If true, shuffle ties (identical years).
                Defaults to True.

        Returns:
            Iterable[int]: sorted indices
        """
        densities = ds.get_years(range(len(ds)))
        # we use these numbers to shuffle the data in case of ties
        random_numbers = np.arange(len(densities))
        if shuffle:
            np.random.shuffle(random_numbers)

        t = [(v, i, random_numbers[i]) for i, v in enumerate(densities)]
        t.sort(reverse=not self.ascending, key=lambda x: (x[0], x[2]))
        indices = [i for _, i, _ in t]

        return indices


class RandomSplitter(Splitter):
    """The "conventional" splitter.

    Random split data into sets/folds"""

    def get_sorted_indices(
        self, ds: StructureDataset, shuffle: Optional[bool] = True
    ) -> Iterable[int]:
        """Simply return a list of indices of length equal to the length of the dataset."""
        indices = np.arange(len(ds))
        if shuffle:
            np.random.shuffle(indices)

        return indices


class FingerprintSplitter(Splitter):
    """Splitter that uses the features of the structures to split the data."""

    def __init__(
        self,
        feature_names: List[str],
    ) -> None:
        self.feature_names = feature_names
        super().__init__()

    def get_sorted_indices(self, ds: StructureDataset, shuffle: Optional[bool] = True):
        """Return a list of indices, where rows are soreted according to
        their similarity (considering the features specified in the class construction)"""
        indices = ds._df.sort_values(by=self.feature_names).index.values
        return indices
