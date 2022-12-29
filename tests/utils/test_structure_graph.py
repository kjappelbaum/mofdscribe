# -*- coding: utf-8 -*-
"""Test helper functions for dealing with structure graphs."""
from mofdscribe.featurizers.utils.structure_graph import get_neighbors_up_to_scope

def test_get_neighbors_up_to_scope(hkust_structure_graph): 
    res = get_neighbors_up_to_scope(hkust_structure_graph, 1, 3)
    assert len(res[1]) == 4
    assert len(res[2]) == 4
    assert len(res[3]) == 8