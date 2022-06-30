from functools import partial
from typing import Type

from pymatgen.core import IMolecule, IStructure, Molecule, Structure


def add_operates_on(cls, type: Type):
    """Adds the `operates_on` classmethod to a featurizer class.

    This is useful for SBUFeaturizers that need to know what type of
    input they need to pass to the featurizer.

    Args:
        cls (type): The class to add the `operates_on` classmethod to.
        types (Type): The types that the featurizer can operate on.

    Returns:
        cls: The class with the `operates_on` classmethod added.
    """
    try:
        cls._accepted_types.append(type)
    except AttributeError:
        setattr(cls, "_accepted_types", [type])

    def operates_on(self):
        return self._accepted_types

    cls.operates_on = operates_on
    return cls


operates_on_structure = partial(add_operates_on, type=Structure)
operates_on_istructure = partial(add_operates_on, type=IStructure)
operates_on_molecule = partial(add_operates_on, type=Molecule)
operates_on_imolecule = partial(add_operates_on, type=IMolecule)
