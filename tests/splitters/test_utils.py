"""Test the splitter helping functions."""

import numpy as np

from mofdscribe.splitters.utils import kennard_stone_sampling


def test_kennard_stone_sampling():
    """Ensure we get the order we would expect."""
    X = np.array([[1, 2, 3], [4, 5, 6], [8, 8, 9]])  # noqa: N806

    indices = kennard_stone_sampling(X)
    assert indices == [2, 0, 1]

    # Make sure also the other options do not complain
    indices = kennard_stone_sampling(X, centrality_measure="median")
    assert indices == [2, 0, 1]

    indices = kennard_stone_sampling(X, centrality_measure="random")
    assert len(indices) == 3  # we cannot guarantee the order of the indices
