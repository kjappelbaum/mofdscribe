from rdkit import Chem

from mofdscribe.featurizers.bu.smarts_matches import SmartsMatchCounter, number_smart_matches


def test_number_smart_matches():
    """Test that we get the correct number of matches."""
    mol = Chem.MolFromSmiles("C1=CC=CC=C1")
    smarts = ["n1nnnc1"]
    assert number_smart_matches(mol, smarts) == 0

    mol = Chem.MolFromSmiles("CCCP=O")
    smarts = ["[C,S,P]=O"]
    assert number_smart_matches(mol, smarts) == 1


def test_smart_match_counter(linker_molecule):
    """Test that we get the correct number of matches."""
    smarts = ["[S,P]=O"]
    featurizer = SmartsMatchCounter(smarts, feature_labels=["acid_groups"])
    assert featurizer.featurize(linker_molecule) == [0]

    aromatic_ring_smarts = ["c1ccccc1"]
    featurizer = SmartsMatchCounter(aromatic_ring_smarts, feature_labels=["aromatic_rings"])
    assert featurizer.featurize(linker_molecule) == [6]
