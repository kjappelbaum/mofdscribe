# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np


class Splitter:
    def k_fold_spliter(self, ds, k, **kwargs):
        """
        Split the data into k folds.
        """
        data = []
        for fold in range(k):
            # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
            # to k-1.
            frac_fold = 1.0 / (k - fold)
            fold_inds = self.split(
                ds, frac_train=frac_fold, frac_valid=1 - frac_fold, frac_test=0, **kwargs
            )

            data.append(ds.select(fold_inds))
        return data

    def train_test_split(self, ds, frac_train, frac_test, **kwargs):
        """
        Split the data into train and test.
        """
        train_inds, test_inds = self.split(
            ds, frac_train=frac_train, frac_valid=0, frac_test=frac_test, **kwargs
        )
        return ds.select(train_inds), ds.select(test_inds)

    def train_valid_test_split(self, ds, frac_train, frac_valid, frac_test, **kwargs):
        """
        Split the data into train, validation and test.
        """
        train_inds, valid_inds, test_inds = self.split(
            ds,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            **kwargs,
        )
        return ds.select(train_inds), ds.select(valid_inds), ds.select(test_inds)

    def split(self, ds, frac_train, frac_valid, frac_test, **kwargs):
        """
        Split the data into train, validation and test.
        """
        raise NotImplementedError


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

    def train_valid_test_split(
        self,
        ds,
        frac_train: float,
        frac_valid: float,
        sample_frac: float = 1.0,
        shuffle: bool = True,
        **kwargs,
    ):
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

        return train_inds, valid_inds, test_inds

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
            yield train_fold, test_fold
            current = stop
