from mofdscribe.chemistry.partialchargestats import PartialChargeStats
import pytest
from tests.conftest import hkust_structure, irmof_structure


def test_partial_charge_stats(hkust_structure, irmof_structure):
    for structure in [hkust_structure, irmof_structure]:
        featurizer = PartialChargeStats()
        feats = featurizer.featurize(structure)
        assert len(feats) == 3
    assert len(featurizer.feature_labels()) == 3
    assert len(featurizer.citations()) == 2
