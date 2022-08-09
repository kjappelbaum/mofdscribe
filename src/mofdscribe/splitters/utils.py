# -*- coding: utf-8 -*-
"""Helper functions for the splitters.

Some of these methods might also be useful for constructing nested cross-validation loops.
"""
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import ArrayLike
from pandas.api.types import infer_dtype
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# todo: can we do here something with numba?
# for numba, we would need to do some rewrite as there is no support
# for numpy.delete
def kennard_stone_sampling(
    X: ArrayLike,  # noqa: N803
    scale: bool = True,
    centrality_measure: str = "mean",
    metric: Union[Callable, str] = "euclidean",
) -> List[int]:
    """Run the Kennard-Stone sampling algorithm [KennardStone].

    The algorithm selects samples with uniform converage.
    The initial samples are biased towards the boundaries of the dataset.

    .. note::

        You also might this algorithm useful for creating a "diverse"
        sample of points, e.g., to initalize an active learnign loop.

    .. warning::

        This algorithm has a high computational complexity.
        It is not recommended for large datasets.

    Args:
        X (ArrayLike): Input feature matrix.
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
            ‘yule’. See :py:meth:`scipy.spatial.distance.cdist`.

            Defaults to "euclidean".

    Raises:
        ValueError: If non-implemented centrality measure is used.

    Returns:
        List[int]: indices sorted by their Max-Min distance.
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # noqa: N806

    original_X = X.copy()  # noqa: N806

    if centrality_measure == "mean":
        distance_to_center = cdist(X, X.mean(axis=0).reshape(1, -1), metric=metric)
    elif centrality_measure == "median":
        distance_to_center = cdist(X, np.median(X, axis=0).reshape(1, -1), metric=metric)
    elif centrality_measure == "random":
        distance_to_center = cdist(X, X[np.random.choice(X.shape[0])].reshape(1, -1), metric=metric)
    else:
        raise ValueError(f"Unknown centrality measure: {centrality_measure}")

    index_farthest = np.argmax(distance_to_center)

    index_selected = [index_farthest]

    index_remaining = np.arange(len(X))

    X = np.delete(X, index_selected, axis=0)  # noqa: N806
    index_remaining = np.delete(index_remaining, index_selected, axis=0)

    # we will sometimes need to do quite a few steps.
    # for all practical datasets this will exceeed python's recursion limit.
    # we therefore use a while loop to avoid this.
    with tqdm(total=len(X) - 1) as pbar:
        while len(index_remaining):
            samples_selected = original_X[index_selected]
            min_distance_to_samples_selected = np.min(cdist(samples_selected, X, metric=metric))

            index_farthest = np.argmax(min_distance_to_samples_selected)

            index_selected.append(index_remaining[index_farthest])
            X = np.delete(X, index_farthest, axis=0)  # noqa: N806
            index_remaining = np.delete(index_remaining, index_farthest, 0)
            pbar.update(1)

    return index_selected


def pca_kmeans(
    X: np.ndarray,  # noqa: N803
    scaled: bool,
    n_pca_components: Union[int, str],
    n_clusters: int,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    pca_kwargs: Optional[Dict[str, Any]] = None,
    kmeans_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Run principal component analysis (PCA) followed by K-means clustering on the data.

    Uses sklearn's implementation of PCA, and k-means.

    Args:
        X (np.ndarray): Input data
        scaled (bool): If True, use standard scaling for clustering
        n_pca_components (Union[int, str]): number of principal components to keep
        n_clusters (int): number of clusters
        random_state (Optional[Union[int, np.random.RandomState]], optional): Random state for sklearn.
            Defaults to None.
        pca_kwargs (Dict[str, Any], optional): Additional keyword arguments for
            sklearn's :py:class:`sklearn.decomposition.PCA`. Defaults to None.
        kmeans_kwargs (Dict[str, Any], optional):  Additional keyword arguments for
            sklearn's :py:class:`sklearn.clustering.KMeans`. Defaults to None.

    Returns:
        np.ndarray: Cluster indices.
    """
    if scaled:
        X = StandardScaler().fit_transform(X)  # noqa: N806

    if n_pca_components is not None:
        pca = PCA(n_components=n_pca_components, **(pca_kwargs or {}))
        X_pca = pca.fit_transform(X)  # noqa: N806
    else:
        X_pca = X  # noqa: N806
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **(kmeans_kwargs or {}))
    return kmeans.fit_predict(X_pca)


def is_categorical(x: Union[float, int, np.typing.ArrayLike]) -> bool:
    """Return true if x is categorial or composed of integers."""
    return infer_dtype(x) == "category" or infer_dtype(x) == "integer"


def stratified_train_test_partition(
    idxs: Iterable[int],
    stratification_col: np.typing.ArrayLike,
    train_size: float,
    valid_size: float,
    test_size: float,
    shuffle: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
) -> Tuple[np.array, np.array, np.array]:
    """Perform a stratified train/test split.

    .. seealso::

        * :py:meth:`mofdscribe.splitters.utils.grouped_stratified_train_test_partition`:
          performs an grouped stratified train/test split
        * :py:meth:`mofdscribe.splitters.utils.grouped_train_valid_test_partition`:
          performs an grouped un-stratified train/test split

    Args:
        idxs (Iterable[int]): Indices of points to split
        stratification_col (np.typing.ArrayLike): Data used for stratification.
            If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
            then we directly use it for stratification. Otherwise, we use quantile binning.
        train_size (float): Size of the training set as fraction.
        valid_size (float): Size of the validation set as fraction.
        test_size (float): Size of the test set as fraction.
        shuffle (bool): If True, perform a shuffled split. Defaults to True.
        random_state (Union[int, np.random.RandomState], optional):
            Random state for the suffler. Defaults to None.
        q (Iterable[float], optional): List of quantiles used for quantile binning.
            Defaults to (0, 0.25, 0.5, 0.75, 1).

    Returns:
        Tuple[np.array, np.array, np.array]: Train, validation, test indices.
    """
    if stratification_col is not None:
        if is_categorical(stratification_col):
            stratification_col = stratification_col
        else:
            logger.warning(
                "Stratifying on non-categorical data. "
                "Note that there is still discussion on the usefullness of this method."
            )

            stratification_col = quantile_binning(stratification_col, q=q)

    train_size, valid_size, test_size = get_train_valid_test_sizes(
        len(stratification_col), train_size, valid_size, test_size
    )

    train_idx, test_idx, train_strat, _ = train_test_split(
        idxs,
        stratification_col,
        train_size=train_size + valid_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratification_col,
    )

    if valid_size > 0:
        train_idx, valid_idx = train_test_split(
            train_idx,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=train_strat,
        )
    else:
        valid_idx = []

    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


def grouped_stratified_train_test_partition(
    stratification_col: np.typing.ArrayLike,
    group_col: np.typing.ArrayLike,
    train_size: float,
    valid_size: float,
    test_size: float,
    shuffle: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    q: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
    center: callable = np.median,
) -> Tuple[np.array, np.array, np.array]:
    """Return grouped stratified train-test partition.

    First, we compute the most common stratification category / centrality measure
    of the stratification column for every group.
    Then, we perform a stratified train/test partition on the groups.
    We then "expand" by concatenating the indices belonging to each group.

    .. warning::

        Note that this won't work well if the number of groups and datapoints is small.
        It will also cause issues if the number of datapoints in the groups is very
        imbalanced.

    .. seealso::

        * :py:meth:`mofdscribe.splitters.utils.stratified_train_test_partition`:
          performs an un-grouped stratified train/test split
        * :py:meth:`mofdscribe.splitters.utils.grouped_train_valid_test_partition`:
          performs an grouped un-stratified train/test split

    Args:
        stratification_col (np.typing.ArrayLike):  Data used for stratification.
            If it is categorical (see :py:meth:`mofdscribe.splitters.utils.is_categorical`)
            then we directly use it for stratification. Otherwise, we use quantile binning.
        group_col (np.typing.ArrayLike): Data used for grouping.
        train_size (float): Size of the training set as fraction.
        valid_size (float): Size of the validation set as fraction.
        test_size (float): Size of the test set as fraction.
        shuffle (bool): If True, perform a shuffled split. Defaults to True.
        random_state (Union[int, np.random.RandomState], optional):
            Random state for the suffler. Defaults to None.
        q (Iterable[float], optional): List of quantiles used for quantile binning.
            Defaults to [0, 0.25, 0.5, 0.75, 1].
        center (callable): Aggregation function to compute a measure of centrality
            of all the points in a group such that this can then be used for stratification.
            This is only used for continuos inputs. For categorical inputs, we always use
            the mode.

    Returns:
        Tuple[np.array, np.array, np.array]: Train, validation, test indices.
    """
    groups = np.unique(group_col)
    categorical = is_categorical(stratification_col)

    category_for_group = []
    # for each group, get the mode
    if categorical:
        for group in groups:
            category_for_group.append(np.mode(stratification_col[group_col == group]))
    else:
        for group in groups:
            category_for_group.append(center(stratification_col[group_col == group]))

    # if we do not have categories, we now need to discretize the stratification_col
    if not categorical:
        category_for_group = quantile_binning(category_for_group, q=q)

    train_size, valid_size, test_size = get_train_valid_test_sizes(
        len(category_for_group), train_size, valid_size, test_size
    )

    # now we can do the split
    train_groups, test_groups, train_cat, _ = train_test_split(
        groups,
        category_for_group,
        train_size=train_size + valid_size,
        test_size=test_size,
        stratify=category_for_group,
        shuffle=shuffle,
        random_state=random_state,
    )

    if valid_size > 0:
        train_groups, valid_groups = train_test_split(
            train_groups,
            train_size=train_size,
            stratify=train_cat,
            shuffle=shuffle,
            random_state=random_state,
        )

    # now, get the original indices
    train_indices = np.where(np.isin(group_col, train_groups))[0]
    if valid_size > 0:
        valid_indices = np.where(np.isin(group_col, valid_groups))[0]
    else:
        valid_indices = []
    test_indices = np.where(np.isin(group_col, test_groups))[0]

    return train_indices, valid_indices, test_indices


def get_train_valid_test_sizes(
    size: int, train_size: float, valid_size: float, test_size: float
) -> Tuple[int, int, int]:
    """Compute the number of points in every split."""
    train_size = int(np.floor(train_size * size))
    valid_size = int(np.ceil(valid_size * size))
    test_size = int(size - train_size - valid_size)
    return train_size, valid_size, test_size


def grouped_train_valid_test_partition(
    groups: np.typing.ArrayLike,
    train_size: float,
    valid_size: float,
    test_size: float,
    shuffle: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[np.array, np.array, np.array]:
    """Perform a grouped train/test split without stratification.

    .. seealso::

        * :py:meth:`mofdscribe.splitters.utils.stratified_train_test_partition`:
          performs an un-grouped stratified train/test split
        * :py:meth:`mofdscribe.splitters.utils.grouped_stratified_train_test_partition`:
          performs an grouped stratified train/test split

    Args:
        groups (np.typing.ArrayLike): Data used for grouping.
        train_size (float): Size of the training set as fraction.
        valid_size (float): Size of the validation set as fraction.
        test_size (float): Size of the test set as fraction.
        shuffle (bool): If True, perform a shuffled split. Defaults to True.
        random_state (Union[int, np.random.RandomState], optional):
            Random state for the suffler. Defaults to None.

    Returns:
        Tuple[np.array, np.array, np.array]: Train, validation, test indices.
    """
    train_indices = []
    valid_indices = []
    test_indices = []

    unique_groups = np.unique(groups)

    train_size, valid_size, test_size = get_train_valid_test_sizes(
        len(unique_groups), train_size, valid_size, test_size
    )

    train_groups, test_groups = train_test_split(
        unique_groups,
        train_size=train_size + valid_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    if valid_size > 0:
        train_groups, valid_groups = train_test_split(
            train_groups,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state,
        )

    # now, get the original indices
    train_indices = np.where(np.isin(groups, train_groups))[0]
    if valid_size > 0:
        valid_indices = np.where(np.isin(groups, valid_groups))[0]
    else:
        valid_indices = None
    test_indices = np.where(np.isin(groups, test_groups))[0]

    return train_indices, valid_indices, test_indices


def quantile_binning(values: np.typing.ArrayLike, q: Iterable[float]) -> np.array:
    """Use :py:meth:`pandas.qcut` to bin the values based on quantiles."""
    values = pd.qcut(values, q, labels=np.arange(len(q) - 1)).astype(int)
    return values


def check_fraction(train_fraction: float, valid_fraction: float, test_fraction: float) -> None:
    """Check that the fractions are all between 0 and 1 and that they sum up to 1."""
    for name, fraction in [
        ("train fraction", train_fraction),
        ("valid fraction", valid_fraction),
        ("test fraction", test_fraction),
    ]:
        if not (fraction <= 1) & (fraction >= 0):
            raise ValueError(
                f"{name} is {fraction}. However, train/valid/test fractions must be between 0 and 1."
            )

    logger.debug(
        f"Using fractions: train: {train_fraction}, valid: {valid_fraction}, test: {test_fraction}"
    )
    if not (train_fraction + valid_fraction + test_fraction) == 1:
        raise ValueError("Train, valid, test fractions must sum to 1.")


def no_group_warn(groups: Optional[np.typing.ArrayLike]) -> None:
    """Raise warning if groups is None."""
    if groups is None:
        logger.warning(
            "You are not using a grouped split."
            " However, for retricular materials, grouping is typically a good idea to avoid data leakage."
        )


def downsample_splits(splits, sample_frac):
    downsampled = []
    for split in splits:
        downsampled.append(np.random.choice(split, int(len(split) * sample_frac)))

    return tuple(downsampled)


def sort_arrays_by_len(arrays, sort=True):
    if sort:
        arrays = [np.array(array) for array in arrays]
        arrays.sort(key=len, reverse=True)
        return tuple(arrays)
    else:
        return tuple(arrays)
