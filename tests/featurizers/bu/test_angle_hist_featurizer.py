# -*- coding: utf-8 -*-

from mofdscribe.featurizers.bu.angle_hist_featurizer import PairWiseAngleHist


def test_angle_hist_featurizer(molecule):
    featurizer = PairWiseAngleHist(density=True)
    feats = featurizer.featurize(molecule)
    labels = featurizer.feature_labels()
    assert len(feats) == len(labels)
    assert feats[0] == 0
