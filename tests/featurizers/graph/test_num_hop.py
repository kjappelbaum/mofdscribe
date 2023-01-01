# -*- coding: utf-8 -*-
import numpy as np

from mofdscribe.featurizers.graph.numhops import NumHops
from mofdscribe.mof import MOF


def test_num_hops(hkust_structure):
    mof = MOF(hkust_structure)
    num_hops = NumHops()
    features = num_hops.featurize(mof, [0, 463])
    assert len(features) == len(num_hops.feature_labels())
    assert np.allclose(features, [1.0, 0, 1.0, 1.0])

    features = num_hops.featurize(mof, [0, 272])
    assert len(features) == len(num_hops.feature_labels())
    assert np.allclose(features, [2.0, 0.0, 2.0, 2.0])
