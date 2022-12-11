# -*- coding: utf-8 -*-
from structuregraph_helpers.create import get_structure_graph
from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from mofdscribe.featurizers.hostguest.utils import remove_guests_from_structure


def test_remove_guests_from_structure(floating_structure):
    """Test the remove_guests_from_structure function."""
    sg = get_structure_graph(floating_structure)

    _, _, idx, _, _ = get_subgraphs_as_molecules(sg)
    structure = remove_guests_from_structure(floating_structure, idx)
    assert len(structure) == len(floating_structure) - 5
