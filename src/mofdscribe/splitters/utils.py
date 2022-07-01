# -*- coding: utf-8 -*-
"""Helper functions for the splitters."""
from typing import Callable, List, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
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
             ‘yule’. Defaults to "euclidean".

    Raises:
        ValueError: If non-implemented centrality measure is used.

    Returns:
        List[int]: indices sorted by their Max-Min distance.

    References:
        [KennardStone] R. W. Kennard & L. A. Stone (1969): Computer Aided Design of Experiments,
            Technometrics, 11:1, 137-148.
            https://www.tandfonline.com/doi/abs/10.1080/00401706.1969.10490666
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
