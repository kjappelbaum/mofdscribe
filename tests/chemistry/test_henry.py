# -*- coding: utf-8 -*-
from mofdscribe.chemistry.henry import Henry

from ..helpers import is_jsonable


def test_henryhoa(hkust_structure, irmof_structure):
    featurizer = Henry(cycles=1000)
    features = featurizer.featurize(hkust_structure)
    labels = featurizer.feature_labels()
    assert len(features) == len(labels)
    features_irmof = featurizer.featurize(irmof_structure)
    assert sum(abs(features - features_irmof)) > 0
    assert is_jsonable(dict(zip(featurizer.feature_labels(), features)))
