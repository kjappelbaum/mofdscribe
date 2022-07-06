"""Make sure that the utils work."""
import pytest
from pymatgen.core import IMolecule, IStructure, Molecule, Structure
from matminer.featurizers.base import BaseFeaturizer
from mofdscribe.featurizers.chemistry import PartialChargeStats
from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)


class TestFeaturizer(BaseFeaturizer):
    def citations(self):
        return []

    def featurize(self, structure):
        return [0]

    def feature_labels(self):
        return ["test"]

    def implementors(self):
        return []


def test_add_operates_on():
    """Test that the add_operates_on decorator works."""

    class TestClass:
        pass

    decorator_test = operates_on_structure(TestClass)

    assert decorator_test().operates_on() == [Structure]

    with pytest.raises(AttributeError):
        TestFeaturizer()._accepted_types

    with pytest.raises(AttributeError):
        TestFeaturizer().operates_on()

    @operates_on_molecule
    class DecoratorTest2(TestFeaturizer):
        pass

    assert DecoratorTest2().operates_on() == [Molecule]

    @operates_on_structure
    @operates_on_molecule
    @operates_on_imolecule
    @operates_on_istructure
    class DecoratorTest3(TestFeaturizer):
        pass

    assert set(DecoratorTest3().operates_on()) == {Structure, Molecule, IMolecule, IStructure}
