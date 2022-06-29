# -*- coding: utf-8 -*-
"""Test the PH image featurizer."""
from mofdscribe.featurizers.topology.ph_image import PHImage

from ..helpers import is_jsonable


def test_phimage(hkust_structure, irmof_structure, cof_structure):
    """Ensure we get the correct number of features."""
    phi = PHImage()
    for structure in [hkust_structure, irmof_structure, cof_structure]:
        features = phi.featurize(structure)
        assert len(features) == 20 * 20 * 4 * 3

    assert len(phi.feature_labels()) == 20 * 20 * 4 * 3
    assert is_jsonable(dict(zip(phi.feature_labels(), features)))
    assert features.ndim == 1


def test_phimage_fit(hkust_structure, irmof_structure):
    """Ensure that calling fit changes the settings."""
    phi = PHImage()
    phi.fit([hkust_structure, irmof_structure])

    assert len(phi.max_b) == len(phi.max_p) == 4
    assert phi.max_b[0] == phi.max_b[3] == 0
    assert 3.0 < phi.max_b[1] < 3.5

    assert 2.6 < phi.max_p[0] < 3.5
