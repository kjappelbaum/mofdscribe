# -*- coding: utf-8 -*-
from mofdscribe.featurizers.bu.angle_stats_featurizer import PairWiseAngleStats


def test_angle_stats_featurizer(molecule):
    featurizer = PairWiseAngleStats()
    feats = featurizer.featurize(molecule)
    labels = featurizer.feature_labels()
    assert len(feats) == len(labels)
    assert (feats[0] > 90) & (feats[0] < 120)
    assert feats[-1] == 180
