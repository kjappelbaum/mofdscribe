# -*- coding: utf-8 -*-
"""Test the RACs featurizer."""
from pymatgen.core import IStructure

from mofdscribe.chemistry._fragment import get_bb_indices
from mofdscribe.chemistry.racs import RACS, _get_racs_for_bbs
from mofdscribe.utils.structure_graph import get_structure_graph

from ..helpers import is_jsonable


def test_racs(hkust_structure, irmof_structure):
    """Make sure that the featurization works for typical MOFs and the number of
    features is as expected.
    """
    for structure in [hkust_structure, irmof_structure]:
        featurizer = RACS()
        feats = featurizer.featurize(structure)
        assert len(feats) == 4 * 3 * 2 * 5  # 4 properties, 3 scopes, 2 aggregations, 5 bb types
        sg = get_structure_graph(IStructure.from_sites(structure), featurizer.bond_heuristic)
        racs = {}
        bb_indices = get_bb_indices(sg)
        for bb in featurizer._bbs:
            racs.update(
                _get_racs_for_bbs(
                    bb_indices[bb],
                    sg,
                    featurizer.attributes,
                    featurizer.scopes,
                    featurizer.prop_agg,
                    featurizer.corr_agg,
                    featurizer.bb_agg,
                    bb,
                )
            )
        assert list(racs.keys()) == featurizer.feature_labels()

    assert len(featurizer.feature_labels()) == 120
    assert len(featurizer.citations()) == 2
    assert is_jsonable(dict(zip(featurizer.feature_labels(), feats)))
    assert feats.ndim == 1
