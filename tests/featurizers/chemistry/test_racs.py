# -*- coding: utf-8 -*-
"""Test the RACs featurizer."""
from collections import OrderedDict

import numpy as np
from pymatgen.core import IStructure

from mofdscribe.featurizers.chemistry._fragment import get_bb_indices
from mofdscribe.featurizers.chemistry.racs import RACS, _get_racs_for_bbs
from mofdscribe.featurizers.utils.structure_graph import get_structure_graph

from ..helpers import is_jsonable


def test_racs(hkust_structure, irmof_structure):
    """Make sure that the featurization works for typical MOFs and the number of features is as expected."""
    for structure in [hkust_structure, irmof_structure]:
        featurizer = RACS()
        feats = featurizer.featurize(structure)
        sg = get_structure_graph(IStructure.from_sites(structure), featurizer.bond_heuristic)
        racs = {}
        bb_indices = get_bb_indices(sg)
        for bb in featurizer._bbs:
            v = _get_racs_for_bbs(
                bb_indices[bb],
                sg,
                featurizer.attributes,
                featurizer.scopes,
                featurizer.prop_agg,
                featurizer.corr_agg,
                featurizer.bb_agg,
                bb,
            )
            racs.update(v)
            # There are no functional groups in those MOFs, so we except nans for this "scope"
            if ("functional" not in bb) and ("linker" in bb):
                assert np.isnan(np.array(list(v.values()))).sum() == 0
            elif "functional" in bb:
                assert np.isnan(np.array(list(v.values()))).sum() == len(v)
        racs_ordered = OrderedDict(sorted(racs.items()))

        assert list(racs_ordered.keys()) == featurizer.feature_labels()

    # assert len(featurizer.feature_labels()) == 120
    assert len(featurizer.citations()) == 2
    assert is_jsonable(dict(zip(featurizer.feature_labels(), feats)))
    assert feats.ndim == 1


def test_racs_functional(irmof_structure, abacuf_structure, floating_structure):
    # ABACUF doesn't have linkers with a core. It is simply HCOO
    for structure in [abacuf_structure]:
        featurizer = RACS(primitive=False)  # because also the linker structure is not primitive
        feats = featurizer.featurize(structure)
        # assert len(feats) == 4 * 3 * 8 * 5  # 4 properties, 3 scopes, 8 aggregations, 5 bb types
        sg = get_structure_graph(IStructure.from_sites(structure), featurizer.bond_heuristic)
        racs = {}
        bb_indices = get_bb_indices(sg)
        for bb in featurizer._bbs:
            v = _get_racs_for_bbs(
                bb_indices[bb],
                sg,
                featurizer.attributes,
                featurizer.scopes,
                featurizer.prop_agg,
                featurizer.corr_agg,
                featurizer.bb_agg,
                bb,
            )
            racs.update(v)
            # we classify the "O" as a functional group
            assert np.isnan(np.array(list(v.values()))).sum() == 0
        racs_ordered = OrderedDict(sorted(racs.items()))
        assert list(racs_ordered.keys()) == featurizer.feature_labels()

    # assert len(featurizer.feature_labels()) == 120
    assert len(featurizer.citations()) == 2
    assert is_jsonable(dict(zip(featurizer.feature_labels(), feats)))
    assert feats.ndim == 1

    floating_feats = featurizer.featurize(floating_structure)
    irmof_feats = featurizer.featurize(irmof_structure)

    assert np.allclose(floating_feats, irmof_feats, rtol=0.05, equal_nan=True)
