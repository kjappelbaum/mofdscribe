"""Make sure that the utils work."""
import pytest
from pymatgen.core import IMolecule, IStructure, Molecule, Structure

from mofdscribe.featurizers.chemistry import PartialChargeStats
from mofdscribe.featurizers.utils.extend import (
    operates_on_imolecule,
    operates_on_istructure,
    operates_on_molecule,
    operates_on_structure,
)


def test_add_operates_on():
    """Test that the add_operates_on decorator works."""

    class TestClass(PartialChargeStats):
        pass

    decorator_test = operates_on_structure(TestClass)

    assert decorator_test().operates_on() == [Structure]

    with pytest.raises(AttributeError):
        PartialChargeStats()._accepted_types

    with pytest.raises(AttributeError):
        PartialChargeStats().operates_on()

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

    assert set(DecoratorTest3().operates_on()) == {Structure, Molecule, IMolecule, IStructure}
