# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np

__all__ = (
    "DensitySplitter",
    "HashSplitter",
    "TimeSplitter",
    "Splitter",
    "RandomSplitter",
    "FingerprintSplitter",
)


class Splitter:
    def train_valid_test_split(
        self,
        ds,
        frac_train: float,
        frac_valid: float,
        sample_frac: float = 1.0,
        shuffle: bool = True,
        **kwargs,
    ):
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
        ds,
        frac_train: float,
        sample_frac: float = 1.0,
        shuffle: bool = True,
        **kwargs,
    ):
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

    def k_fold(self, ds, k, shuffle: bool = True):
        """
        Split the data into k folds.
        """
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
    def __init__(
        self,
        hash_type: str = "undecorated_scaffold_hash",
    ) -> None:
        self.hash_type = hash_type
        super().__init__()

    def get_hashes(self, ds):
        number_of_points = len(ds)
        if self.hash_type == "undecorated_scaffold_hash":
            hashes = ds.get_undecorated_scaffold_hashes(range(number_of_points))
        elif self.hash_type == "decorated_graph_hash":
            hashes = ds.get_decorated_graph_hashes(range(number_of_points))
        elif self.hash_type == "decorated_scaffold_hash":
            hashes = ds.get_decorated_scaffold_hashes(range(number_of_points))
        elif self.hash_type == "undecorated_graph_hash":
            hashes = ds.get_undecorated_graph_hash(range(number_of_points))
        else:
            raise ValueError(f"Unknown hash type: {self.hash_type}")

        return hashes

    def get_sorted_indices(self, ds, shuffle: bool = True):
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
    def __init__(
        self,
        ascending: bool = True,
    ) -> None:
        self.ascending = ascending
        super().__init__()

    def get_sorted_indices(self, ds, shuffle: bool = True):
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
    def __init__(
        self,
        ascending: bool = True,
    ) -> None:
        self.ascending = ascending
        super().__init__()

    def get_sorted_indices(self, ds, shuffle: bool = True):
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
    def get_sorted_indices(self, ds, shuffle: bool = True):
        indices = np.arange(len(ds))
        if shuffle:
            np.random.shuffle(indices)

        return indices


class FingerprintSplitter(Splitter):
    def __init__(
        self,
        feature_names,
    ) -> None:
        self.feature_names = feature_names
        super().__init__()

    def get_sorted_indices(self, ds, shuffle: bool = True):
        indices = ds._df.sort_values(by=self.feature_names).index.values
        return indices
