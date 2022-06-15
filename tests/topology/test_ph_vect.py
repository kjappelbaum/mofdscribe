# -*- coding: utf-8 -*-
"""Test the PH Vect featurizer"""
import pytest

from mofdscribe.topology.ph_vect import PHVect

from ..helpers import is_jsonable


def test_ph_vect(hkust_structure, irmof_structure):
    """Ensure we get the correct number of features and that they are different for different structures"""
    # should raise if not fitted
    with pytest.raises(ValueError):
        ph_vect = PHVect()
        ph_vect.featurize(hkust_structure)

    # should be able to fit and featurize
    ph_vect = PHVect(n_components=2)
    ph_vect.fit([hkust_structure, irmof_structure])

    # test fit_transform
    ph_vect = PHVect(n_components=2)
    feat = ph_vect.fit_transform([hkust_structure, irmof_structure])
    assert feat.shape == (2, 4 * 2 * 2)
    assert is_jsonable(dict(zip(ph_vect.feature_labels(), feat[0])))
    assert feat.ndim == 2

    feat = ph_vect.featurize(hkust_structure)
    assert feat.ndim == 1

    assert len(feat) == len(set(ph_vect.feature_labels()))

    assert len(feat) == 4 * 2 * 2
