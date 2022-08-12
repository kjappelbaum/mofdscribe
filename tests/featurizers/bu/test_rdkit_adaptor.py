# -*- coding: utf-8 -*-
"""Test the RDkitAdaptor."""
import numpy as np
from rdkit.Chem.Descriptors3D import Asphericity

from mofdscribe.featurizers.bu.rdkitadaptor import RDKitAdaptor


def test_rdkit_adaptor(molecule_graph, molecule):
    """Test that we can call RDKit featurizers with pymatgen molecules."""
    adaptor = RDKitAdaptor(Asphericity, ["asphericity"], "vesta")

    features_a = adaptor.featurize(molecule)
    assert features_a.shape == (1,)
    assert features_a.dtype == np.float

    features_b = adaptor.featurize(molecule_graph)
    assert all(features_a == features_b)
