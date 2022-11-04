# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS, MA_ARRAY_AGGREGATORS


def test_aggregators():
    test_array = [1, 2, 3, 4, 5, 6, 7]
    masked_test_array = np.ma.array(
        test_array + [8], mask=[False, False, False, False, False, False, False, True]
    )

    true_values = {
        "sum": 28,
        "avg": 4,
        "max": 7,
        "min": 1,
        "range": 6,
        "std": 2,
        "mean": 4,
        "median": 4,
        "harmean": 2.6997245179063363,
        "geomean": 3.3800151591412964,
        "mad": 2.0,
        "trimean": 4.0,
        "inf": 7,
        "manhattan": 28,
    }

    for aggregator_name, aggregator in ARRAY_AGGREGATORS.items():
        masked_aggregator = MA_ARRAY_AGGREGATORS[aggregator_name]
        assert aggregator(test_array) == pytest.approx(true_values[aggregator_name])
        if aggregator_name == "inf":
            assert masked_aggregator(masked_test_array) == pytest.approx(8)
        elif aggregator_name == "manhattan":
            assert masked_aggregator(masked_test_array) == pytest.approx(36)
        else:
            assert masked_aggregator(masked_test_array) == pytest.approx(
                true_values[aggregator_name]
            )
