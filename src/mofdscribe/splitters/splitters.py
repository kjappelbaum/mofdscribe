# -*- coding: utf-8 -*-
"""Classes that help performing cross-validation."""
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


from loguru import logger

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedGroupKFold
import pandas as pd
from .utils import (
    is_categorical,
    kennard_stone_sampling,
    pca_kmeans,
    grouped_stratified_train_test_partition,
    stratified_train_test_partition,
    grouped_train_valid_test_partition,
    quantile_binning,
)
from ..datasets.dataset import StructureDataset

__all__ = (
    "DensitySplitter",
    "HashSplitter",
    "TimeSplitter",
    "Splitter",
    "RandomSplitter",
    "FingerprintSplitter",
    "KennardStoneSplitter",
    "ClusterSplitter",
)


def no_group_warn(groups):
    if groups is None:
        logger.warning(
            "You are not using a grouped split. However, this is typically a good idea to avoid data leakage."
        )


class BaseSplitter:
    def __init__(
        self,
        ds,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = None,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center=np.median,
        q=[0, 0.25, 0.5, 0.75, 1],
    ):
        self.ds = ds
        self.shuffle = shuffle
        self.random_state = random_state
        self._len = len(ds)
        self._sample_frac = sample_frac
        self._stratification_col = stratification_col
        self._center = center
        self._q = q

    def _get_idxs(self):
        return np.random.choice(
            np.arange(self._len), int(self._len * self._sample_frac), replace=False
        )

    def train_test_split(self, train_size: float = 0.7) -> Tuple[Iterable[int], Iterable[int]]:
        groups = self._get_groups()
        stratification_col = self._get_stratification_col()
        idx = self._get_idxs()
        no_group_warn(groups)
        if groups is not None:
            if stratification_col is not None:
                train_idx, _, test_index = grouped_stratified_train_test_partition(
                    stratification_col[idx],
                    groups[idx],
                    train_size,
                    0,
                    1 - train_size,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                    center=self._center,
                    q=self._q,
                )
            else:
                train_idx, _, test_index = grouped_train_valid_test_partition(
                    groups[idx],
                    train_size,
                    0,
                    1 - train_size,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )

        else:
            stratification_col = stratification_col[idx] if stratification_col is not None else None
            train_idx, _, test_index = stratified_train_test_partition(
                self._get_idxs(),
                stratification_col,
                train_size=train_size,
                shuffle=self.shuffle,
                random_state=self.random_state,
                q=self._q,
            )

        return train_idx, test_index

    def train_valid_test_split(
        self, train_size: float = 0.7, valid_size: float = 0.1
    ) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
        groups = self._get_groups()
        stratification_col = self._get_stratification_col()
        idx = self._get_idxs()
        no_group_warn(groups)

        if groups is not None:
            if stratification_col is not None:
                train_idx, valid_idx, test_index = grouped_stratified_train_test_partition(
                    stratification_col[idx],
                    groups[idx],
                    train_size,
                    valid_size,
                    1 - train_size - valid_size,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                    center=self._center,
                    q=self._q,
                )
            else:
                train_idx, valid_idx, test_index = grouped_train_valid_test_partition(
                    groups[idx],
                    train_size,
                    valid_size,
                    1 - train_size - valid_size,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
        else:
            stratification_col = stratification_col[idx] if stratification_col is not None else None
            train_idx, _, test_index = stratified_train_test_partition(
                self._get_idxs(),
                stratification_col,
                train_size=train_size,
                shuffle=self.shuffle,
                random_state=self.random_state,
                q=self._q,
            )
        return train_idx, valid_idx, test_index

    def kfold_split(self, n_splits=5) -> Tuple[Iterable[int], Iterable[int]]:
        groups = self._get_groups()
        stratification_col = self._get_stratification_col()
        no_group_warn(groups)
        idx = self._get_idxs()

        groups = groups[idx] if groups is not None else None
        stratification_col = stratification_col[idx] if stratification_col is not None else None

        if not is_categorical(stratification_col):
            stratification_col = quantile_binning(stratification_col, self._q)
        kfold = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        for train_index, test_index in kfold.split(idx, stratification_col, groups=groups):
            yield train_index, test_index

    def _get_groups(self) -> Iterable[Union[int, str]]:
        return None

    def _get_stratification_col(self) -> Iterable[Union[int, float]]:
        if isinstance(self._stratification_col, str):
            return self.ds[self._stratification_col].values
        else:
            return self._stratification_col


class HashSplitter(BaseSplitter):
    """Splitter that uses graph hashes to split the data in more stringent ways.

    Note that the hashes we use do not allow for a meaningful measure of
    similarity.

    However, we can sort the data by the hash and make sure that the same hash
    is only occuring in one set. Moreover, we take care that the largest groups
    of duplicated hashes are in the training set.
    """

    def __init__(
        self,
        ds,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = None,
        hash_type: str = "undecorated_scaffold_hash",
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center=np.median,
        q=[0, 0.25, 0.5, 0.75, 1],
    ) -> None:
        """Initialize a HashSplitter.

        Args:
            hash_type (str): Hash type to use. Must be one of the
                following:
                * undecorated_scaffold_hash
                * decorated_graph_hash
                * decorated_scaffold_hash
                * undecorated_graph_hash
                Defaults to "undecorated_scaffold_hash".
        """
        self.hash_type = hash_type
        super().__init__(ds, shuffle, random_state, sample_frac, stratification_col, center, q)

    def _get_hashes(self, ds: StructureDataset) -> Iterable[str]:
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

    def _get_groups(self) -> Iterable[int]:
        return self.get_hashes(self.ds)


class DensitySplitter(BaseSplitter):
    """Splitter that uses the density of the structures to split the data.

    For this, we sort structures according to their density and then group the based on the density.
    You can modify the number of groups using the :attr:`density_q` parameter, those values indicate
    the quantiles which we use for the grouping.

    This ensures that the validation is quite stringent as the different folds will have different densities.
    """

    def __init__(
        self,
        ds,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = None,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center=np.median,
        q=[0, 0.25, 0.5, 0.75, 1],
        density_q=[0, 0.25, 0.5, 0.75, 1],
    ) -> None:
        """Initialize the DensitySplitter class."""
        self._density_q = density_q
        super().__init__(ds, shuffle, random_state, sample_frac, stratification_col, center, q)

    def _get_groups(self) -> Iterable[int]:
        return quantile_binning(self.ds.get_density(), self._density_q)


class TimeSplitter(BaseSplitter):
    """This splitter sorts structures according to their publication date.

    That is, the training set will contain structures that are "older" (have
    been discovered earlier) than the ones in the test set.
    """

    def __init__(
        self,
        ds,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = None,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center=np.median,
        q=[0, 0.25, 0.5, 0.75, 1],
        year_q=[0, 0.25, 0.5, 0.75, 1],
    ) -> None:
        """Initialize the TimeSplitter class"""
        self._year_q = year_q
        super().__init__(ds, shuffle, random_state, sample_frac, stratification_col, center, q)

    def _get_groups(self) -> Iterable[int]:
        return quantile_binning(self._ds.get_years(range(len(self._ds))), self._year_q)


class FingerprintSplitter(BaseSplitter):
    """Splitter that uses the features of the structures to split the data.

    It does not directly compute distances but simply *sorts* the rows
    using `np.sort`.
    For distance-based splits, see :py:meth:`~mofdscribe.splitters.KennardStoneSplitter`.
    """

    def __init__(
        self,
        feature_names: List[str],
    ) -> None:
        """Construct a FingerPrintSplitter.

        Args:
            feature_names (List[str]): Names of features to consider.
        """
        self.feature_names = feature_names
        super().__init__()

    def get_sorted_indices(self, ds: StructureDataset, shuffle: bool = True) -> Iterable[int]:
        """Return a list of indices, sorted by similarity.

        Here, rows are sorted according to
        their similarity (considering the features specified in the class construction)

        Args:
            ds (StructureDataset): A mofdscribe StructureDataset
            shuffle (bool): Not used in this method.
                Defaults to True.

        Returns:
            Iterable[int]: Sorted indices.
        """
        indices = ds._df.sort_values(by=self.feature_names).index.values
        return indices


class KennardStoneSplitter(BaseSplitter):
    """Run the Kennard-Stone sampling algorithm [KennardStone].

    The algorithm selects samples with uniform converage.
    The initial samples are biased towards the boundaries of the dataset.

    .. warning::

        This splitter can be slow for large datasets as
        it requires us to perform distance matrices N times for a dataset
        with N structures.
    """

    def __init__(
        self,
        feature_names: List[str],
        scale: bool = True,
        centrality_measure: str = "mean",
        metric: Union[Callable, str] = "euclidean",
        ascending: bool = False,
    ) -> None:
        """Construct a KennardStoneSplitter.

        Args:
            feature_names (List[str]): Names of features to consider.
            scale (bool): If True, apply z-score normalization
                prior to running the sampling. Defaults to True.
            centrality_measure (str): The first sample is selected to be
                maximally distanct from this value. It can be one of "mean", "median",
                "random". In case of "random" we simply select a random point.
                In the case of "mean" and "median" the initial point is maximally distanct
                from the mean and median of the feature matrix, respectively.
                Defaults to "mean".
            metric (Union[Callable, str]): The distance metric to use.
                If a string, the distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’,
                ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
                ‘jensenshannon’, ‘kulsinski’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
                ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’,
                ‘yule’. Defaults to "euclidean".
            ascending (bool): If True, sort samples in asceding distance to the center.
                That is, the first samples (maximally distant to center) would be sampled last.
                Defaults to False.
        """
        self.feature_names = feature_names
        self.scale = scale
        self.centrality_measure = centrality_measure
        self.metric = metric
        self.ascending = ascending
        self._sorted_indices = None
        super().__init__()

    def get_sorted_indices(self, ds: StructureDataset, shuffle: bool = True) -> Iterable[int]:
        """Return a list of indices, sorted by similarity using the Kennard-Stone algorithm.

        The first sample will be maximally distant from the center.

        Args:
            ds (StructureDataset): A mofdscribe StructureDataset
            shuffle (bool): Not used in this method.
                Defaults to True.

        Returns:
            Iterable[int]: Sorted indices.
        """
        if self._sorted_indices is None:
            feats = ds._df[self.feature_names].values

            indices = kennard_stone_sampling(
                feats,
                scale=self.scale,
                centrality_measure=self.centrality_measure,
                metric=self.metric,
            )

            if self.ascending:
                indices = indices[::-1]

            self._sorted_indices = indices
        return self._sorted_indices


# ToDo: n_pca_components could also be "auto" based on elbow criterion
class ClusterSplitter(BaseSplitter):
    """Split the data into clusters and keep clusters in different folds (as much as possible).

    The approach has been proposed on
    `Kaggle <https://www.kaggle.com/code/lucamassaron/are-you-doing-cross-validation-the-best-way/notebook>`_.
    In principle, we perform the following steps:

    1. Scale the data (optional).
    2. Perform PCA for de-correlation.
    3. Perform k-means clustering.
    """

    def __init__(
        self,
        feature_names: List[str],
        scaled: bool = True,
        n_pca_components: Optional[int] = "mle",
        n_clusters: int = 4,
        random_state: int = 42,
        pca_kwargs: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a ClusterSplitter.

        Args:
            feature_names (List[str]): Names of features to consider.
            scaled (bool): If True, scale the data before clustering.
                Defaults to True.
            n_pca_components (Optional[int]): Number of components to use for PCA.
                If "mle", use the number of components that maximizes the variance.
                Defaults to "mle".
            n_clusters (int): Number of clusters to use.
                Defaults to 4.
            random_state (int): Random seed.
                Defaults to 42.
            pca_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to PCA.
                Defaults to None.
            kmeans_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to k-means.
                Defaults to None.
        """
        self.feature_names = feature_names
        self.scaled = scaled
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._sorted_indices = None
        self.ascending = False
        self._pca_kwargs = pca_kwargs
        self._kmeans_kwargs = kmeans_kwargs
        super().__init__()

    def get_sorted_indices(self, ds: StructureDataset, shuffle: bool = True) -> Iterable[int]:
        if self._sorted_indices is None:
            feats = ds._df[self.feature_names].values

            clusters = pca_kmeans(
                feats,
                n_clusters=self.n_clusters,
                n_pca_components=self.n_pca_components,
                random_state=self.random_state,
                scaled=self.scaled,
                pca_kwargs=self._pca_kwargs,
                kmeans_kwargs=self._kmeans_kwargs,
            )
            random_numbers = np.arange(len(clusters))
            if shuffle:
                np.random.shuffle(random_numbers)

            t = [(v, i, random_numbers[i]) for i, v in enumerate(clusters)]
            t.sort(reverse=not self.ascending, key=lambda x: (x[0], x[2]))
            indices = [i for _, i, _ in t]
            self._sorted_indices = indices
        return self._sorted_indices


class ClusterStratifiedSplitter(BaseSplitter):
    """Split the data into clusters and stratify on those clusters

    The approach has been proposed on
    `Kaggle <https://www.kaggle.com/code/lucamassaron/are-you-doing-cross-validation-the-best-way/notebook>`_.
    In principle, we perform the following steps:

    1. Scale the data (optional).
    2. Perform PCA for de-correlation.
    3. Perform k-means clustering.
    """

    def __init__(
        self,
        feature_names: List[str],
        scaled: bool = True,
        n_pca_components: Optional[int] = "mle",
        n_clusters: int = 4,
        random_state: int = 42,
        pca_kwargs: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a ClusterStratifiedSplitter.

        Args:
            feature_names (List[str]): Names of features to consider.
            scaled (bool): If True, scale the data before clustering.
                Defaults to True.
            n_pca_components (Optional[int]): Number of components to use for PCA.
                If "mle", use the number of components that maximizes the variance.
                Defaults to "mle".
            n_clusters (int): Number of clusters to use.
                Defaults to 4.
            random_state (int): Random seed.
                Defaults to 42.
            pca_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to PCA.
                Defaults to None.
            kmeans_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to k-means.
                Defaults to None.
        """
        self.feature_names = feature_names
        self.scaled = scaled
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._stratification_groups = None
        self.ascending = False
        self._pca_kwargs = pca_kwargs
        self._kmeans_kwargs = kmeans_kwargs
        super().__init__()

    def _get_stratification_groups(
        self, ds: StructureDataset, shuffle: bool = True
    ) -> Iterable[int]:
        if self._stratification_groups is None:
            feats = ds._df[self.feature_names].values

            clusters = pca_kmeans(
                feats,
                n_clusters=self.n_clusters,
                n_pca_components=self.n_pca_components,
                random_state=self.random_state,
                scaled=self.scaled,
                pca_kwargs=self._pca_kwargs,
                kmeans_kwargs=self._kmeans_kwargs,
            )
            self._stratification_groups = clusters
        return self._stratification_groups


class LOCOCV(BaseSplitter):
    """
    Leave-one-cluster-out cross-validation.

    The general idea has been discussed before, e.g. in [Kramer]_.
    Perhaps more widely used in the materials community is [Meredig]_.
    Here, we perform PCA, followed by k-means clustering.

    * Where k = 2 for a train/test split
    * Where k = 3 for a train/valid/test split
    * Where k = k for k-fold crossvalidation

    By default, we will sort outputs such that the cluster sizes are
    train >= test >= valid.

    References:
        [Kramer] `Kramer, C.; Gedeck, P. Leave-Cluster-Out Cross-Validation
            Is Appropriate for Scoring Functions Derived from Diverse Protein Data Sets.
            Journal of Chemical Information and Modeling, 2010, 50, 1961–1969.
            <https://doi.org/10.1021/ci100264e>`_

        [Meredig] `Meredig, B.; Antono, E.; Church, C.; Hutchinson, M.; Ling, J.; Paradiso,
            S.; Blaiszik, B.; Foster, I.; Gibbons, B.; Hattrick-Simpers, J.; Mehta, A.; Ward, L.
            Can Machine Learning Identify the next High-Temperature Superconductor?
            Examining Extrapolation Performance for Materials Discovery.
            Molecular Systems Design &amp; Engineering, 2018, 3, 819–825.
            <https://doi.org/10.1039/c8me00012c>`_
    """

    def __init__(
        self,
        feature_names: List[str],
        scaled: bool = True,
        n_pca_components: Optional[int] = "mle",
        random_state: int = 42,
        pca_kwargs: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a LOCOCV.

        Args:
            feature_names (List[str]): Names of features to consider.
            scaled (bool): If True, scale the data before clustering.
                Defaults to True.
            n_pca_components (Optional[int]): Number of components to use for PCA.
                If "mle", use the number of components that maximizes the variance.
                Defaults to "mle".
            random_state (int): Random seed.
                Defaults to 42.
            pca_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to PCA.
                Defaults to None.
            kmeans_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to k-means.
                Defaults to None.
        """
        self.scaled = scaled
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self._pca_kwargs = pca_kwargs
        self._kmeans_kwargs = kmeans_kwargs
        self._stratification_groups = None
        self.ascending = False
        self.feature_names = feature_names

    def train_test_split(
        self,
        ds: StructureDataset,
        sample_frac: float = 1,
        shuffle: bool = True,
    ) -> Tuple[Iterable[int], Iterable[int]]:
        groups = pca_kmeans(
            ds._df[self.feature_names].values,
            scaled=self.scaled,
            n_pca_components=self.n_pca_components,
            n_clusters=2,
            random_state=self.random_state,
            pca_kwargs=self._pca_kwargs,
            kmeans_kwargs=self._kmeans_kwargs,
        )

        first_group = np.where(groups == 0)[0]
        second_group = np.where(groups == 1)[0]

        if shuffle:
            np.random.shuffle(first_group)
            np.random.shuffle(second_group)

        # potential downsampling after shuffle
        first_group = first_group[: int(sample_frac * len(first_group))]
        second_group = second_group[: int(sample_frac * len(second_group))]

        if len(first_group) > len(second_group):
            return first_group, second_group

        return second_group, first_group

    def train_valid_test_split(
        self,
        ds: StructureDataset,
        sample_frac: float = 1,
        shuffle: bool = True,
    ) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
        groups = pca_kmeans(
            ds._df[self.feature_names].values,
            scaled=self.scaled,
            n_pca_components=self.n_pca_components,
            n_clusters=3,
            random_state=self.random_state,
            pca_kwargs=self._pca_kwargs,
            kmeans_kwargs=self._kmeans_kwargs,
        )

        first_group = np.where(groups == 0)[0]
        second_group = np.where(groups == 1)[0]
        third_group = np.where(groups == 2)[0]

        if shuffle:
            np.random.shuffle(first_group)
            np.random.shuffle(second_group)
            np.random.shuffle(third_group)

        # potential downsampling after shuffle
        first_group = first_group[: int(sample_frac * len(first_group))]
        second_group = second_group[: int(sample_frac * len(second_group))]
        third_group = third_group[: int(sample_frac * len(third_group))]

        groups_sorted_by_len = sorted(
            [first_group, second_group, third_group], key=len, reverse=True
        )
        return groups_sorted_by_len[0], groups_sorted_by_len[2], groups_sorted_by_len[1]

    def k_fold(self, ds: StructureDataset, k: int, shuffle: bool = True, sample_frac: float = 1):
        groups = pca_kmeans(
            ds._df[self.feature_names].values,
            scaled=self.scaled,
            n_pca_components=self.n_pca_components,
            n_clusters=k,
            random_state=self.random_state,
            pca_kwargs=self._pca_kwargs,
            kmeans_kwargs=self._kmeans_kwargs,
        )

        for group in range(k):
            train = np.where(groups != group)[0]
            test = np.where(groups == group)[0]
            if shuffle:
                np.random.shuffle(train)
                np.random.shuffle(test)
            # potential downsampling after shuffle
            train = train[: int(sample_frac * len(train))]
            test = test[: int(sample_frac * len(test))]
            if len(train) > len(test):
                yield train, test
            else:
                yield test, train
