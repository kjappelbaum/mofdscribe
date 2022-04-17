from pymatgen.core import Structure
from mofdscribe.utils.substructures import get_metal_indices, select_elements


def test_get_metal_indices(irmof_structure):
    s = irmof_structure
    metal_indices = get_metal_indices(s)
    assert len(metal_indices) == 32
    assert metal_indices[0] == 0
    assert metal_indices[-1] == 31


def test_select_elements(irmof_structure):
    s = irmof_structure
    s = select_elements(s, "Zn")
    assert isinstance(s, Structure)
    assert len(s) == 32
    assert s[0].specie.symbol == "Zn"
