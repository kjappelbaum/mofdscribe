# -*- coding: utf-8 -*-
from mofdscribe.topology.ph_stats import PHStats


def test_ph_stats(hkust_structure, irmof_structure):
    for structure in [hkust_structure, irmof_structure]:
        featurizer = PHStats()
        features = featurizer.featurize(structure)
        feature_labels = featurizer.feature_labels()
        assert len(features) == len(feature_labels)

    hkust_feats = featurizer.featurize(hkust_structure)
    irmof_feats = featurizer.featurize(irmof_structure)
    assert (hkust_feats != irmof_feats).any()
