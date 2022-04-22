# -*- coding: utf-8 -*-
from mofdscribe.utils.structure_graph import get_neighbors_at_distance


def test_get_neighbors_at_distance(hkust_structure_graph):

    assert len(get_neighbors_at_distance(hkust_structure_graph, 1, 1)[1]) == 5

    assert len(get_neighbors_at_distance(hkust_structure_graph, 1, 2)[1]) == 8
