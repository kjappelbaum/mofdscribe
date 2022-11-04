# -*- coding: utf-8 -*-
"""Classes that help performing cross-validation.

Our splitters attempt to reduce any potential for data leakage by using grouping by default--
and prioritizing grouping over stratficiation or exactly matching the requested train test ratio.

See also the `sklearn docs
<https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data>`_.

.. warning::

    Due to the grouping operations, the train/test ratios the methods produce will not exactly
    match the one you requested.
    For this reason, please get the length of the train/test/valid indices the methods produce.
"""
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from .utils import (
    check_fraction,
    downsample_splits,
    grouped_stratified_train_test_partition,
    grouped_train_valid_test_partition,
    is_categorical,
    kennard_stone_sampling,
    no_group_warn,
    pca_kmeans,
    quantile_binning,
    sort_arrays_by_len,
    stratified_train_test_partition,
)
from ..datasets.dataset import AbstractStructureDataset

__all__ = (
    "DensitySplitter",
    "HashSplitter",
    "TimeSplitter",
    "BaseSplitter",
    "KennardStoneSplitter",
    "ClusterSplitter",
    "LOCOCV",
    "ClusterStratifiedSplitter",
)


class BaseSplitter:
    """A :code:`BaseSplitter` implements the basic logic for dataset partition as well as k-fold cross-validation.

    Methods that inherit from this class typically implement the

        * :code: `_get_stratification_col`: Should return an ArrayLike object of floats, categories, or ints.
            If it is categorical data, the :code:`BaseSplitter` will handle the discretization.
        * :code: `_get_groups`: Should return an ArrayLike object of categories (integers or strings)

    methods.
    Internally, the :code:`BaseSplitter` uses those to group and/or stratify the splits.
    """

    # Those variables are needed to automatically set the number of groups
    # if the users does not set them (default behavior).
    _grouping_q = None
    _set_grouping = False

    def __init__(
        self,
        ds: AbstractStructureDataset,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center: callable = np.median,
        q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
        sort_by_len: bool = True,
    ):
        """Initialize a BaseSplitter.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Optional[Union[int, np.random.RandomState]], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (Optional[float], optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            stratification_col (Union[str, np.typing.ArrayLike], optional): Data used for stratification.
                If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
                then we directly use it for stratification. Otherwise, we use quantile binning.
                Defaults to None.
            center (callable): Aggregation function to compute a measure of centrality
                of all the points in a group such that this can then be used for stratification.
                This is only used for continuos inputs. For categorical inputs, we always use
                the mode. Defaults to np.median.
            q (Iterable[float], optional): List of quantiles used for quantile binning.
                Defaults to (0, 0.25, 0.5, 0.75, 1).
            sort_by_len (bool): If True, sort the splits by length.
                (Applies to the train/test/valid and train/test splits). Defaults to True.
        """
        self._ds = ds
        self._shuffle = shuffle
        self._random_state = random_state
        self._len = len(ds)
        self._sample_frac = sample_frac
        self._stratification_col = stratification_col
        self._center = center
        self._q = q
        self._sort_by_len = sort_by_len

        logger.debug(
            f"Splitter settings | shuffle {self._shuffle}, "
            f"random state {self._random_state}, sample frac {self._sample_frac}, q {self._q}"
        )

    def _get_idxs(self):
        """Return an array of indices. Length equals to the length of the dataset."""
        idx = np.arange(self._len)
        return idx

    def train_test_split(self, frac_train: float = 0.7) -> Tuple[Iterable[int], Iterable[int]]:
        """Perform a train/test partition.

        Args:
            frac_train (float): Fraction of the data to use for the training set.
                Defaults to 0.7.

        Returns:
            Tuple[Iterable[int], Iterable[int]]: Train indices, test indices
        """
        if self._grouping_q is None or self._set_grouping:
            self._set_grouping = True
            self._grouping_q = np.linspace(0, 1, 3)
        check_fraction(train_fraction=frac_train, valid_fraction=0, test_fraction=1 - frac_train)
        groups = self._get_groups()
        stratification_col = self._get_stratification_col()
        idx = self._get_idxs()
        no_group_warn(groups)
        if groups is not None:
            if stratification_col is not None:
                logger.debug("Using grouped stratified partition")
                train_idx, _, test_index = grouped_stratified_train_test_partition(
                    stratification_col[idx],
                    groups[idx],
                    frac_train,
                    0,
                    1 - frac_train,
                    shuffle=self._shuffle,
                    random_state=self._random_state,
                    center=self._center,
                    q=self._q,
                )
            else:
                logger.debug("Using grouped partition")
                train_idx, _, test_index = grouped_train_valid_test_partition(
                    groups[idx],
                    frac_train,
                    0,
                    1 - frac_train,
                    shuffle=self._shuffle,
                    random_state=self._random_state,
                )

        else:
            stratification_col = stratification_col[idx] if stratification_col is not None else None
            logger.debug("Using stratified partition")
            train_idx, _, test_index = stratified_train_test_partition(
                self._get_idxs(),
                stratification_col,
                train_size=frac_train,
                valid_size=0,
                test_size=1 - frac_train,
                shuffle=self._shuffle,
                random_state=self._random_state,
                q=self._q,
            )

        if self._sample_frac < 1:
            return sort_arrays_by_len(
                downsample_splits([train_idx, test_index], self._sample_frac), self._sort_by_len
            )
        return sort_arrays_by_len([train_idx, test_index], self._sort_by_len)

    def train_valid_test_split(
        self, frac_train: float = 0.7, frac_valid: float = 0.1
    ) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
        """Perform a train/valid/test partition.

        Args:
            frac_train (float): Fraction of data to use for the training set.
                Defaults to 0.7.
            frac_valid (float): Fraction of data to use for the validation set.
                Defaults to 0.1.

        Returns:
            Tuple[Iterable[int], Iterable[int], Iterable[int]]: Training, validation, test set.
        """
        if self._grouping_q is None or self._set_grouping:
            self._set_grouping = True
            self._grouping_q = np.linspace(0, 1, 4)
        check_fraction(
            train_fraction=frac_train,
            valid_fraction=frac_valid,
            test_fraction=1 - frac_train - frac_valid,
        )
        groups = self._get_groups()
        stratification_col = self._get_stratification_col()
        idx = self._get_idxs()
        no_group_warn(groups)

        if groups is not None:
            if stratification_col is not None:
                logger.debug("Using grouped stratified partition")
                train_idx, valid_idx, test_index = grouped_stratified_train_test_partition(
                    stratification_col[idx],
                    groups[idx],
                    frac_train,
                    frac_valid,
                    1 - frac_train - frac_valid,
                    shuffle=self._shuffle,
                    random_state=self._random_state,
                    center=self._center,
                    q=self._q,
                )
            else:
                logger.debug("Using grouped  partition")
                train_idx, valid_idx, test_index = grouped_train_valid_test_partition(
                    groups[idx],
                    frac_train,
                    frac_valid,
                    1 - frac_train - frac_valid,
                    shuffle=self._shuffle,
                    random_state=self._random_state,
                )
        else:
            logger.debug("Using stratified partition")
            stratification_col = stratification_col[idx] if stratification_col is not None else None
            train_idx, valid_idx, test_index = stratified_train_test_partition(
                self._get_idxs(),
                stratification_col,
                train_size=frac_train,
                valid_size=frac_valid,
                test_size=1 - frac_train - frac_valid,
                shuffle=self._shuffle,
                random_state=self._random_state,
                q=self._q,
            )
        if self._sample_frac < 1:
            return sort_arrays_by_len(
                downsample_splits([train_idx, valid_idx, test_index], self._sample_frac),
                self._sort_by_len,
            )

        return sort_arrays_by_len([train_idx, valid_idx, test_index], self._sort_by_len)

    def k_fold(self, k: int = 5) -> Tuple[Iterable[int], Iterable[int]]:
        """Peform k-fold crossvalidation.

        Args:
            k (int): Number of folds. Defaults to 5.

        Yields:
            Tuple[Iterable[int], Iterable[int]]: Train indices, test indices.
        """
        if self._grouping_q is None or self._set_grouping:
            self._set_grouping = True
            self._grouping_q = np.linspace(0, 1, k + 1)
        groups = self._get_groups()
        stratification_col = self._get_stratification_col()
        no_group_warn(groups)
        idx = self._get_idxs()

        groups = groups[idx] if groups is not None else None
        stratification_col = stratification_col[idx] if stratification_col is not None else None

        if stratification_col is not None:
            if not is_categorical(stratification_col):
                stratification_col = quantile_binning(stratification_col, self._q)

            if groups is not None:
                kfold = StratifiedGroupKFold(
                    n_splits=k, shuffle=self._shuffle, random_state=self._random_state
                )
            else:
                kfold = StratifiedKFold(
                    n_splits=k, shuffle=self._shuffle, random_state=self._random_state
                )

        else:
            # this is not shuffled?
            if groups is not None:
                kfold = GroupKFold(n_splits=k)
            else:
                kfold = KFold(n_splits=k)

        if groups is not None:
            for train_index, test_index in kfold.split(idx, y=stratification_col, groups=groups):
                if self._shuffle:
                    np.random.shuffle(train_index)
                    np.random.shuffle(test_index)
                if self._sample_frac < 1:
                    yield downsample_splits([train_index, test_index], self._sample_frac)
                else:
                    yield train_index, test_index
        else:
            for train_index, test_index in kfold.split(idx, y=stratification_col):
                if self._shuffle:
                    np.random.shuffle(train_index)
                    np.random.shuffle(test_index)
                if self._sample_frac < 1:
                    yield downsample_splits([train_index, test_index], self._sample_frac)
                else:
                    yield train_index, test_index

    def _get_groups(self) -> Iterable[Union[int, str]]:
        return None

    def _get_stratification_col(self) -> Iterable[Union[int, float]]:
        if isinstance(self._stratification_col, str):
            return self._ds._df[self._stratification_col].values
        else:
            return self._stratification_col


class HashSplitter(BaseSplitter):
    """Splitter that uses Weisfeiller-Lehman graph hashes [WL]_ to split the data in more stringent ways.

    Note that the hashes we use do not allow for a meaningful measure of
    similarity. That is, there is no way to measure the distance between two strings.
    The only meaningful measure is if they are identical or not.


    .. note::

        Weisfeiller-Lehman graph hashes do not give a guarantee for graph-isomorphism.
        That is, there might be identical hashes that do not correspond to isomorphic graphs.

    .. note::

        There are certain graphs that a Weisfeiller-Lehman test cannot distinguish [Bouritsas]_.

    .. note::

        We speak about Weisfeiller-Lehman hashes as they are the defaults for the mofdscribe datasets.
        However, you can also overwrite this method with a custom hashing function.
    """

    def __init__(
        self,
        ds: AbstractStructureDataset,
        hash_type: str = "undecorated_scaffold_hash",
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center=np.median,
        q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
        sort_by_len: bool = True,
    ) -> None:
        """Initialize a HashSplitter.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            hash_type (str): Hash type to use. Must be one of the
                following:
                * undecorated_scaffold_hash
                * decorated_graph_hash
                * decorated_scaffold_hash
                * undecorated_graph_hash
                Defaults to "undecorated_scaffold_hash".
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            stratification_col (Union[str, np.typing.ArrayLike], optional): Data used for stratification.
                If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
                then we directly use it for stratification. Otherwise, we use quantile binning.
                Defaults to None.
            center (callable, optional): Aggregation function to compute a measure of centrality
                of all the points in a group such that this can then be used for stratification.
                This is only used for continuos inputs. For categorical inputs, we always use
                the mode. Defaults to np.median.
            q (Iterable[float], optional): List of quantiles used for quantile binning.
                Defaults to (0, 0.25, 0.5, 0.75, 1).
            sort_by_len (bool): If True, sort the splits by length.
                (Applies to the train/test/valid and train/test splits). Defaults to True.
        """
        self.hash_type = hash_type
        super().__init__(
            ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            stratification_col=stratification_col,
            center=center,
            q=q,
            sort_by_len=sort_by_len,
        )

    def _get_hashes(self) -> Iterable[str]:
        """Retrieve the list of hashes from the dataset

        Raises:
            ValueError: If the hash type is not one of the following:
                * undecorated_scaffold_hash
                * decorated_graph_hash
                * decorated_scaffold_hash
                * undecorated_graph_hash

        Returns:
            Iterable[str]: list of hashes
        """
        number_of_points = len(self._ds)
        if self.hash_type == "undecorated_scaffold_hash":
            hashes = self._ds.get_undecorated_scaffold_hashes(range(number_of_points))
        elif self.hash_type == "decorated_graph_hash":
            hashes = self._ds.get_decorated_graph_hashes(range(number_of_points))
        elif self.hash_type == "decorated_scaffold_hash":
            hashes = self._ds.get_decorated_scaffold_hashes(range(number_of_points))
        elif self.hash_type == "undecorated_graph_hash":
            hashes = self._ds.get_undecorated_graph_hashes(range(number_of_points))
        else:
            raise ValueError(f"Unknown hash type: {self.hash_type}")

        return hashes

    def _get_groups(self) -> Iterable[int]:
        return self._get_hashes()


class DensitySplitter(BaseSplitter):
    """Splitter that uses the density of the structures to split the data.

    For this, we sort structures according to their density and then group the based on the density.
    You can modify the number of groups using the :attr:`density_q` parameter, those values indicate
    the quantiles which we use for the grouping.

    This ensures that the validation is quite stringent as the different folds will have different densities.

    The motivations for doing this are:

        * density is often one of the most important descriptors for gas uptake properties.

        * there is often is a very large difference in density distribution
            between hypothetical and experimental databases.
    """

    def __init__(
        self,
        ds: AbstractStructureDataset,
        density_q: Optional[Iterable[float]] = None,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center: callable = np.median,
        q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
        sort_by_len: bool = True,
    ) -> None:
        """Initialize the DensitySplitter class.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            density_q (Iterable[float], optional): List of quantiles used for quantile binning for the density.
                Defaults to None. If None, then we use two bins for test/train split, three for
                validation/train/test split and k for k-fold.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            stratification_col (Union[str, np.typing.ArrayLike], optional): Data used for stratification.
                If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
                then we directly use it for stratification. Otherwise, we use quantile binning.
                Defaults to None.
            center (callable): Aggregation function to compute a measure of centrality
                of all the points in a group such that this can then be used for stratification.
                This is only used for continuos inputs. For categorical inputs, we always use
                the mode. Defaults to np.median.
            q (Iterable[float], optional): List of quantiles used for quantile binning.
                Defaults to (0, 0.25, 0.5, 0.75, 1]. Defaults to [0, 0.25, 0.5, 0.75, 1).
            sort_by_len (bool): If True, sort the splits by length.
                (Applies to the train/test/valid and train/test splits). Defaults to True.
        """
        self._grouping_q = density_q
        super().__init__(
            ds=ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            stratification_col=stratification_col,
            center=center,
            q=q,
            sort_by_len=sort_by_len,
        )

    def _get_groups(self) -> Iterable[int]:
        return quantile_binning(self._ds.get_densities(range(len(self._ds))), self._grouping_q)


class TimeSplitter(BaseSplitter):
    """This splitter sorts structures according to their publication date.

    That is, the training set will contain structures that are "older" (have
    been discovered earlier) than the ones in the test set.
    This can mimick real-life model development conditions [MoleculeNet]_.

    It has for instance also be used with ICSD data in [Palizhati]_
    and been the focus of [Sheridan]_.

    .. seealso:

        * The `mp-time-split <https://github.com/sparks-baird/mp-time-split>`_ package
        provides similar functionality for data from the materials project.
    """

    def __init__(
        self,
        ds: AbstractStructureDataset,
        year_q: Optional[Iterable[float]] = None,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center: callable = np.median,
        q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
        sort_by_len: bool = True,
    ) -> None:
        """Initialize the TimeSplitter class.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            year_q (Iterable[float]): List of quantiles used for quantile binning on the years.
                Defaults to None. If None, then we use two bins for test/train split, three for
                validation/train/test split and k for k-fold.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            stratification_col (Union[str, np.typing.ArrayLike], optional): Data used for stratification.
                If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
                then we directly use it for stratification. Otherwise, we use quantile binning.
                Defaults to None.
            center (callable): Aggregation function to compute a measure of centrality
                of all the points in a group such that this can then be used for stratification.
                This is only used for continuos inputs. For categorical inputs, we always use
                the mode. Defaults to np.median.
            q (Iterable[float], optional): List of quantiles used for quantile binning.
                Defaults to (0, 0.25, 0.5, 0.75, 1).
            sort_by_len (bool): If True, sort the splits by length.
                (Applies to the train/test/valid and train/test splits). Defaults to True.
        """
        self._grouping_q = year_q
        super().__init__(
            ds=ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            stratification_col=stratification_col,
            center=center,
            q=q,
            sort_by_len=sort_by_len,
        )

    def _get_groups(self) -> Iterable[int]:
        return quantile_binning(self._ds.get_years(range(len(self._ds))), self._grouping_q)


class KennardStoneSplitter(BaseSplitter):
    """Run the Kennard-Stone sampling algorithm [KennardStone]_.

    The algorithm selects samples with uniform converage.
    The initial samples are biased towards the boundaries of the dataset.
    Hence, it might be biased by outliers.

    This algorithm ensures a flat coverage of the dataset.
    It is also known as CADEX algorithm and has been later refined
    in the DUPLEX algorithm [Snee]_.

    .. warning::

        This splitter can be slow for large datasets as
        it requires us to perform distance matrices N times for a dataset
        with N structures.


    .. warning::

        Stratification is not supported for this splitter.


    .. warning::

        I couldn't find a good reference for the k-fold version of
        this algorihm.
    """

    def __init__(
        self,
        ds: AbstractStructureDataset,
        feature_names: List[str],
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        scale: bool = True,
        centrality_measure: str = "mean",
        metric: Union[Callable, str] = "euclidean",
        ascending: bool = False,
    ) -> None:
        """Construct a KennardStoneSplitter.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            feature_names (List[str]): Names of features to consider.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
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
        super().__init__(
            ds=ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            center=None,
            stratification_col=None,
            q=None,
        )

    def get_sorted_indices(self, ds: AbstractStructureDataset) -> Iterable[int]:
        """Return a list of indices, sorted by similarity using the Kennard-Stone algorithm.

        The first sample will be maximally distant from the center.

        Args:
            ds (AbstractStructureDataset): A mofdscribe AbstractStructureDataset

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

    def train_test_split(self, frac_train: float = 0.7) -> Tuple[Iterable[int], Iterable[int]]:
        num_train_points = int(frac_train * len(self._ds))

        if self._shuffle:
            return (
                np.random.permutation(self.get_sorted_indices(self._ds)[:num_train_points]),
                np.random.permutation(self.get_sorted_indices(self._ds)[num_train_points:]),
            )
        return (
            self.get_sorted_indices(self._ds)[:num_train_points],
            self.get_sorted_indices(self._ds)[num_train_points:],
        )

    def train_valid_test_split(
        self, frac_train: float = 0.7, frac_valid: float = 0.1
    ) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
        num_train_points = int(frac_train * len(self._ds))
        num_valid_points = int(frac_valid * len(self._ds))

        if self._shuffle:
            return (
                np.random.permutation(self.get_sorted_indices(self._ds)[:num_train_points]),
                np.random.permutation(
                    self.get_sorted_indices(self._ds)[
                        num_train_points : num_train_points + num_valid_points
                    ]
                ),
                np.random.permutation(
                    self.get_sorted_indices(self._ds)[num_train_points + num_valid_points :]
                ),
            )
        return (
            self.get_sorted_indices(self._ds)[:num_train_points],
            self.get_sorted_indices(self._ds)[
                num_train_points : num_train_points + num_valid_points
            ],
            self.get_sorted_indices(self._ds)[num_train_points + num_valid_points :],
        )

    def k_fold(self, k=5) -> Tuple[Iterable[int], Iterable[int]]:
        kf = KFold(n_splits=k, shuffle=False, random_state=self._random_state)
        for train_index, test_index in kf.split(self.get_sorted_indices(self._ds)):
            if self._shuffle:
                train_index = np.random.permutation(train_index)
                test_index = np.random.permutation(test_index)
            yield train_index, test_index


class ClusterSplitter(BaseSplitter):
    """Split the data into clusters and use the clusters as groups.

    The approach has been proposed on
    `Kaggle <https://www.kaggle.com/code/lucamassaron/are-you-doing-cross-validation-the-best-way/notebook>`_.
    In principle, we perform the following steps:

        1. Scale the data (optional).
        2. Perform PCA for de-correlation.
        3. Perform k-means clustering.
    """

    def __init__(
        self,
        ds: AbstractStructureDataset,
        feature_names: List[str],
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
        center: callable = np.median,
        q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
        sort_by_len: bool = False,
        scaled: bool = True,
        n_pca_components: Optional[Union[int, str]] = "mle",
        n_clusters: int = 4,
        pca_kwargs: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a ClusterSplitter.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            feature_names (List[str]): Names of features to consider.
                        shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            stratification_col (Union[str, np.typing.ArrayLike], optional): Data used for stratification.
                If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
                then we directly use it for stratification. Otherwise, we use quantile binning.
                Defaults to None.
            center (callable): Aggregation function to compute a measure of centrality
                of all the points in a group such that this can then be used for stratification.
                This is only used for continuos inputs. For categorical inputs, we always use
                the mode. Defaults to np.median.
            q (Iterable[float], optional): List of quantiles used for quantile binning.
                Defaults to (0, 0.25, 0.5, 0.75, 1]. Defaults to [0, 0.25, 0.5, 0.75, 1).
            sort_by_len (bool): If True, sort the splits by length.
                (Applies to the train/test/valid and train/test splits). Defaults to True.
            scaled (bool): If True, scale the data before clustering.
                Defaults to True.
            n_pca_components (Union[int, str]): Number of components to use for PCA.
                If "mle", use the number of components that maximizes the variance.
                Defaults to "mle".
            n_clusters (int): Number of clusters to use.
                Defaults to 4.
            random_state (int): Random seed.
                Defaults to 42.
            pca_kwargs (Dict[str, Any]): Keyword arguments to pass to PCA.
                Defaults to None.
            kmeans_kwargs (Dict[str, Any]): Keyword arguments to pass to k-means.
                Defaults to None.
        """
        self.feature_names = feature_names
        self.scaled = scaled
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self._random_state = random_state
        self._sorted_indices = None
        self.ascending = False
        self._pca_kwargs = pca_kwargs
        self._kmeans_kwargs = kmeans_kwargs
        super().__init__(
            ds=ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            stratification_col=stratification_col,
            center=center,
            q=q,
            sort_by_len=sort_by_len,
        )

    def _get_sorted_indices(
        self, ds: AbstractStructureDataset, shuffle: bool = True
    ) -> Iterable[int]:
        if self._sorted_indices is None:
            feats = ds._df[self.feature_names].values

            clusters = pca_kmeans(
                feats,
                n_clusters=self.n_clusters,
                n_pca_components=self.n_pca_components,
                random_state=self._random_state,
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

    def _get_groups(self) -> Iterable[Union[int, str]]:
        si = self._get_sorted_indices(self._ds, self._shuffle)
        return np.array(si)


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
        ds: AbstractStructureDataset,
        feature_names: List[str],
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        scaled: bool = True,
        n_pca_components: Optional[int] = "mle",
        n_clusters: int = 4,
        pca_kwargs: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a ClusterStratifiedSplitter.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            feature_names (List[str]): Names of features to consider.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            scaled (bool): If True, scale the data before clustering.
                Defaults to True.
            n_pca_components (int): Number of components to use for PCA.
                If "mle", use the number of components that maximizes the variance.
                Defaults to "mle".
            n_clusters (int): Number of clusters to use.
                Defaults to 4.
            random_state (int): Random seed.
                Defaults to 42.
            pca_kwargs (Dict[str, Any]): Keyword arguments to pass to PCA.
                Defaults to None.
            kmeans_kwargs (Dict[str, Any]): Keyword arguments to pass to k-means.
                Defaults to None.
        """
        self.feature_names = feature_names
        self.scaled = scaled
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self._random_state = random_state
        self._stratification_groups = None
        self.ascending = False
        self._pca_kwargs = pca_kwargs
        self._kmeans_kwargs = kmeans_kwargs
        super().__init__(
            ds=ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            stratification_col=None,
            center=None,
            q=None,
        )

    def _get_stratification_col(self) -> Iterable[int]:
        if self._stratification_groups is None:
            feats = self._ds._df[self.feature_names].values

            clusters = pca_kmeans(
                feats,
                n_clusters=self.n_clusters,
                n_pca_components=self.n_pca_components,
                random_state=self._random_state,
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
    """

    def __init__(
        self,
        ds: AbstractStructureDataset,
        feature_names: List[str],
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sample_frac: Optional[float] = 1.0,
        scaled: bool = True,
        n_pca_components: Optional[int] = "mle",
        pca_kwargs: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a LOCOCV.

        Args:
            ds (AbstractStructureDataset): A structure dataset.
                The :code:`BaseSplitter` only requires the length magic method to be implemented.
                However, other splitters might require additional methods.
            feature_names (List[str]): Names of features to consider.
            shuffle (bool): If True, perform a shuffled split.
                Defaults to True.
            random_state (Union[int, np.random.RandomState], optional):
                Random state for the shuffling. Defaults to None.
            sample_frac (float, optional):
                This can be used for downsampling. It will randomly select a subset of
                indices from all indices *before* splittings. For instance :code:`sample_frac=0.8`
                will randomly select 80% of the indices before splitting.
                Defaults to 1.0.
            scaled (bool): If True, scale the data before clustering.
                Defaults to True.
            n_pca_components (int): Number of components to use for PCA.
                If "mle", use the number of components that maximizes the variance.
                Defaults to "mle".
            random_state (int): Random seed.
                Defaults to 42.
            pca_kwargs (Dict[str, Any], optional): Additional keyword arguments for
                sklearn's :py:class:`sklearn.decomposition.PCA`. Defaults to None.
            kmeans_kwargs (Dict[str, Any], optional):  Additional keyword arguments for
                sklearn's :py:class:`sklearn.clustering.KMeans`. Defaults to None.
        """
        self.scaled = scaled
        self.n_pca_components = n_pca_components
        self._random_state = random_state
        self._pca_kwargs = pca_kwargs
        self._kmeans_kwargs = kmeans_kwargs
        self._stratification_groups = None
        self.ascending = False
        self.feature_names = feature_names
        super().__init__(
            ds=ds,
            shuffle=shuffle,
            random_state=random_state,
            sample_frac=sample_frac,
            center=None,
            stratification_col=None,
            q=None,
        )

    def train_test_split(
        self,
    ) -> Tuple[Iterable[int], Iterable[int]]:
        """Perform a train/test partition.

        Returns:
            Tuple[Iterable[int], Iterable[int]]: Train indices, test indices
        """
        groups = pca_kmeans(
            self._ds._df[self.feature_names].values,
            scaled=self.scaled,
            n_pca_components=self.n_pca_components,
            n_clusters=2,
            random_state=self._random_state,
            pca_kwargs=self._pca_kwargs,
            kmeans_kwargs=self._kmeans_kwargs,
        )

        first_group = np.where(groups == 0)[0]
        second_group = np.where(groups == 1)[0]

        if self._shuffle:
            np.random.shuffle(first_group)
            np.random.shuffle(second_group)

        # potential downsampling after shuffle
        first_group = first_group[: int(self._sample_frac * len(first_group))]
        second_group = second_group[: int(self._sample_frac * len(second_group))]

        if len(first_group) > len(second_group):
            return first_group, second_group

        return second_group, first_group

    def train_valid_test_split(
        self,
    ) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
        """Perform a train/valid/test partition.

        Returns:
            Tuple[Iterable[int], Iterable[int], Iterable[int]]: Training, validation, test set.
        """
        groups = pca_kmeans(
            self._ds._df[self.feature_names].values,
            scaled=self.scaled,
            n_pca_components=self.n_pca_components,
            n_clusters=3,
            random_state=self._random_state,
            pca_kwargs=self._pca_kwargs,
            kmeans_kwargs=self._kmeans_kwargs,
        )

        first_group = np.where(groups == 0)[0]
        second_group = np.where(groups == 1)[0]
        third_group = np.where(groups == 2)[0]

        if self._shuffle:
            np.random.shuffle(first_group)
            np.random.shuffle(second_group)
            np.random.shuffle(third_group)

        # potential downsampling after shuffle
        first_group = first_group[: int(self._sample_frac * len(first_group))]
        second_group = second_group[: int(self._sample_frac * len(second_group))]
        third_group = third_group[: int(self._sample_frac * len(third_group))]

        groups_sorted_by_len = sorted(
            [first_group, second_group, third_group], key=len, reverse=True
        )
        return groups_sorted_by_len[0], groups_sorted_by_len[2], groups_sorted_by_len[1]

    def k_fold(self, k: int) -> Tuple[Iterable[int], Iterable[int]]:
        """Peform k-fold crossvalidation.

        Args:
            k (int): Number of folds. Defaults to 5.

        Yields:
            Iterator[Tuple[Iterable[int], Iterable[int]]]: Train indices, test indices.
        """
        groups = pca_kmeans(
            self._ds._df[self.feature_names].values,
            scaled=self.scaled,
            n_pca_components=self.n_pca_components,
            n_clusters=k,
            random_state=self._random_state,
            pca_kwargs=self._pca_kwargs,
            kmeans_kwargs=self._kmeans_kwargs,
        )

        for group in range(k):
            train = np.where(groups != group)[0]
            test = np.where(groups == group)[0]
            if self._shuffle:
                np.random.shuffle(train)
                np.random.shuffle(test)
            # potential downsampling after shuffle
            train = train[: int(self._sample_frac * len(train))]
            test = test[: int(self._sample_frac * len(test))]
            if len(train) > len(test):
                yield train, test
            else:
                yield test, train
