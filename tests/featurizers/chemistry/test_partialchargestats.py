# -*- coding: utf-8 -*-
"""Test the PartialChargeStats featurizer."""
from mofdscribe.featurizers.chemistry.partialchargestats import PartialChargeStats

from ..helpers import is_jsonable


def test_partial_charge_stats(hkust_structure, irmof_structure):
    """Make sure that the featurization works for typical MOFs and the number of
    features is as expected.
    """
    for structure in [hkust_structure, irmof_structure]:
        featurizer = PartialChargeStats()
        feats = featurizer.featurize(structure)
        assert len(feats) == 3
    assert len(featurizer.feature_labels()) == 3
    assert len(featurizer.citations()) == 2
    assert is_jsonable(dict(zip(featurizer.feature_labels(), feats)))
    assert feats.ndim == 1
