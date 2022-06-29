# -*- coding: utf-8 -*-
"""Test the AMD featurizer."""
import os

import amd
import pytest

from mofdscribe.featurizers.chemistry.amd import AMD

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_amd_consistency(hkust_structure):
    """Test that AMD is consistent with the original package."""
    amd_mofdscribe = AMD(k=100, atom_types=None)
    amd_hkust_mofdscribe = amd_mofdscribe.featurize(hkust_structure)

    reader = amd.CifReader(os.path.join(THIS_DIR, "..", "..", "test_files", "HKUST-1.cif"))
    amds = [amd.AMD(crystal, 100) for crystal in reader]
    for feat, amd_feat in zip(amds[0], amd_hkust_mofdscribe):
        assert feat == pytest.approx(amd_feat, abs=1e-3)

    amd_mofdscribe = AMD(k=100, aggregations=("mean", "std"))
    amd_hkust_mofdscribe = amd_mofdscribe.featurize(hkust_structure)
    assert len(amd_hkust_mofdscribe) == 100 * 2 * 4
    assert len(amd_mofdscribe.feature_labels()) == 100 * 2 * 4
