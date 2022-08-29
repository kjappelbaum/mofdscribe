# -*- coding: utf-8 -*-
"""Test the PH image featurizer."""
import pytest

from mofdscribe.featurizers.topology.ph_image import PHImage

from ..helpers import is_jsonable


def test_phimage(hkust_structure, irmof_structure, cof_structure, hkust_la_structure):
    """Ensure we get the correct number of features."""
    phi = PHImage()
    for structure in [hkust_structure, irmof_structure, cof_structure]:
        features = phi.featurize(structure)
        assert len(features) == 20 * 20 * 4 * 3

    assert len(phi.feature_labels()) == 20 * 20 * 4 * 3
    assert is_jsonable(dict(zip(phi.feature_labels(), features)))
    assert features.ndim == 1

    # now, try that encoding chemistry with varying atomic radii works
    # we do this by computing the PH images for structures with same geometry
    # and connectivity, but different atoms
    phi = PHImage(
        atom_types=None, alpha_weight="atomic_radius_calculated", min_size=50, max_b=30, max_p=30
    )
    image_cu = phi.featurize(hkust_structure)
    image_la = phi.featurize(hkust_la_structure)
    assert image_cu.shape == image_la.shape
    assert (image_cu != image_la).any()

    # now, to be sure do not encode the same thing with the atomic radius
    phi = PHImage(atom_types=None, alpha_weight=None, min_size=50, max_b=30, max_p=30)
    image_cu = phi.featurize(hkust_structure)
    image_la = phi.featurize(hkust_la_structure)
    assert image_cu.shape == image_la.shape
    assert image_cu == pytest.approx(image_la, rel=1e-2)


def test_phimage_fit(hkust_structure, irmof_structure):
    """Ensure that calling fit changes the settings."""
    phi = PHImage()
    phi.fit([hkust_structure, irmof_structure])

    assert len(phi.max_b) == len(phi.max_p) == 4
    assert phi.max_b[0] == phi.max_b[3] == 0
    assert 3.0 < phi.max_b[1] < 3.5

    assert 2.6 < phi.max_p[0] < 3.5
