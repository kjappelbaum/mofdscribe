"""Make sure that the utils work."""
from mofdscribe.featurizers.chemistry import PartialChargeStats
from mofdscribe.featurizers.utils.extend import (
    operates_on_structure,
    operates_on_molecule,
    operates_on_imolecule,
    operates_on_istructure,
)
from pymatgen.core import Structure, IStructure, Molecule, IMolecule
import pytest


def test_add_operates_on():
    """Test that the add_operates_on decorator works."""

    class TestClass(PartialChargeStats):
        pass

    DecoratorTest = operates_on_structure(TestClass)

    assert DecoratorTest().operates_on() == [Structure]

    with pytest.raises(AttributeError):
        PartialChargeStats()._accepted_types

    @operates_on_molecule
    class DecoratorTest2(PartialChargeStats):
        pass

    assert DecoratorTest2().operates_on() == [Molecule]

    @operates_on_structure
    @operates_on_molecule
    @operates_on_imolecule
    @operates_on_istructure
    class DecoratorTest3(PartialChargeStats):
        pass

    print(DecoratorTest3().operates_on())
    assert set(DecoratorTest3().operates_on()) == {Structure, Molecule, IMolecule, IStructure}
