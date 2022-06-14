# -*- coding: utf-8 -*-
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
    def split(self, ds, frac_train, frac_valid, frac_test, **kwargs):
        """
        Split the data into train, validation and test.
        """
        train_inds = ds.get_indices_for_hash(frac_train)
        valid_inds = ds.get_indices_for_hash(frac_valid)
