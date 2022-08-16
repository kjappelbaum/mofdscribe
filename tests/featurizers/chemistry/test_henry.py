# -*- coding: utf-8 -*-
"""Test the Henry featurizer."""
from mofdscribe.featurizers.chemistry.henry import Henry

from ..helpers import is_jsonable


def test_henryhoa(hkust_structure, irmof_structure):
    """Make sure that the featurization works for typical MOFs and the number of features is as expected."""
    featurizer = Henry(cycles=10)
    features = featurizer.featurize(hkust_structure)
    labels = featurizer.feature_labels()
    assert len(features) == len(labels)
    features_irmof = featurizer.featurize(irmof_structure)
    assert sum(abs(features - features_irmof)) > 0
    assert is_jsonable(dict(zip(featurizer.feature_labels(), features)))
    assert features_irmof.ndim == 1

    featurizer = Henry(cycles=500, return_std=True)
    features = featurizer.featurize(hkust_structure)
    assert len(features) == len(featurizer.feature_labels()) == 4

    # make sure we indeed use pyeqeq
    featurizer = Henry(cycles=500, return_std=True, run_eqeq=False)
    features_no_charge = featurizer.featurize(hkust_structure)
    assert features_no_charge[0] < features[0]
