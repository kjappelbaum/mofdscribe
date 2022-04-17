import pytest
from pymatgen.core import Structure
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def hkust_structure():
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))


@pytest.fixture
def irmof_structure():
    return Structure.from_file(os.path.join(THIS_DIR, "test_files", "IRMOF-1.cif"))
