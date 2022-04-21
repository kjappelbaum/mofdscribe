# -*- coding: utf-8 -*-
import pytest

from mofdscribe.topology.ph_vect import PHVect


def test_ph_vect(hkust_structure, irmof_structure):
    # should raise if not fitted
    with pytest.raises(ValueError):
        ph_vect = PHVect()
        ph_vect.featurize(hkust_structure)

    # should be able to fit and featurize
    ph_vect = PHVect(n_components=2)
    ph_vect.fit([hkust_structure, irmof_structure])

    feat = ph_vect.featurize(hkust_structure)
    assert feat.shape[1] == len(ph_vect.feature_labels())

    assert feat.shape[1] == 4 * 2 * 2

    # test fit_transform
    ph_vect = PHVect(n_components=2)
    feat = ph_vect.fit_transform([hkust_structure, irmof_structure])
    assert feat.shape == (2, 4 * 2 * 2)
