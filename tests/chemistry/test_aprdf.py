# -*- coding: utf-8 -*-
from mofdscribe.chemistry.aprdf import APRDF


def test_aprdf(hkust_structure):
    aprdf_featurizer = APRDF()
    feats = aprdf_featurizer.featurize(hkust_structure)
    assert len(feats) == 1080
    assert len(aprdf_featurizer.feature_labels()) == 1080
