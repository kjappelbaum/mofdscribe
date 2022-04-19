from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import Structure
from mofdscribe.utils.structure_graph import get_structure_graph

# Using MEGNet seems to be the most efficient way to do this.
class ElectronicMEGNet(BaseFeaturizer):
    ...
