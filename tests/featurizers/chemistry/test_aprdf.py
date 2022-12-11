# -*- coding: utf-8 -*-
"""Test APRDF featurizer."""
from mofdscribe.featurizers.chemistry.aprdf import APRDF

from ..helpers import is_jsonable


def test_aprdf(hkust_structure):
    """Make sure that the featurization works for typical MOFs and the number of features is as expected."""
    aprdf_featurizer = APRDF()
    feats = aprdf_featurizer.featurize(hkust_structure)
    assert len(feats) == 432
    assert len(aprdf_featurizer.feature_labels()) == 432
    assert is_jsonable(dict(zip(aprdf_featurizer.feature_labels(), feats)))
