import pytest

from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS


def test_aggregators():
    assert ARRAY_AGGREGATORS["harmmean"]([1, 2, 3, 4, 5, 6, 7]) == pytest.approx(2.6997245179063363)

    assert ARRAY_AGGREGATORS["geomean"]([1, 2, 3, 4, 5, 6, 7]) == pytest.approx(3.3800151591412964)
