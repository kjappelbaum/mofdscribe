# -*- coding: utf-8 -*-
from mofdscribe.pore.voxelgrid import VoxelGrid


def test_voxelgrid(hkust_structure):
    vg = VoxelGrid()
    fl = vg.feature_labels()
    features = vg.featurize(hkust_structure)
    assert len(fl) == len(features)
