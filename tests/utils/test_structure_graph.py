# -*- coding: utf-8 -*-
"""Test helper functions for dealing with structure graphs."""
from mofdscribe.featurizers.utils.structure_graph import get_neighbors_at_distance


def test_get_neighbors_at_distance(hkust_structure_graph):
    """Test the get_neighbors_at_distance function
    at some case studies (manually verified)."""
    assert len(get_neighbors_at_distance(hkust_structure_graph, 1, 1)[1]) == 4

    assert len(get_neighbors_at_distance(hkust_structure_graph, 1, 2)[1]) == 4
