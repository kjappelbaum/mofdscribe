# -*- coding: utf-8 -*-
"""Test the VoxelGrid featurizer."""
from mofdscribe.featurizers.pore.voxelgrid import VoxelGrid
from mofdscribe.mof import MOF


def test_voxelgrid(hkust_structure):
    """Ensure we get the correct number of features."""
    vg = VoxelGrid()
    fl = vg.feature_labels()
    features = vg.featurize(MOF(hkust_structure))
    assert len(fl) == len(features)
