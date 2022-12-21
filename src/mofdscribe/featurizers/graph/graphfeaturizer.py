from functools import lru_cache
from typing import Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.extend import (
    operates_on_istructure,
    operates_on_structure,
    operates_on_structuregraph,
)
from mofdscribe.featurizers.utils.structure_graph import (
    get_structure_graph as get_structure_graph_cached,
)

# operates on needs to be extended to graphs


def get_structure_graph(
    structure: Union[Structure, IStructure], method: str = "vesta"
) -> StructureGraph:
    if isinstance(structure, Structure):
        return get_structure_graph_cached(IStructure.from_sites(structure.sites), method)
    elif isinstance(structure, IStructure):
        return get_structure_graph_cached(structure, method)
    else:
        raise TypeError(f"Expected Structure or IStructure, got {type(structure)}")


# cannot use here the MOFBaseFeaturizer because the input is not always a structure
# not super sure about this yet, but I think, over the long run, I'd like this to only accept graphs
# i'd like to have some kind of transformer objects that handle the conversion from structure to graph
# if needed
# all of this could then be orchestrated in a pipeline object
@operates_on_structuregraph
@operates_on_structure
@operates_on_istructure
class GraphFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()

    def _featurize(self, structure_graph: StructureGraph) -> np.ndarray:
        raise NotImplementedError()

    def featurize(
        self, structure_or_structuregraph: Union[Structure, StructureGraph]
    ) -> np.ndarray:
        if isinstance(structure_or_structuregraph, (Structure, IStructure)):
            structure_graph = get_structure_graph(structure_or_structuregraph)
        elif isinstance(structure_or_structuregraph, StructureGraph):
            structure_graph = structure_or_structuregraph
        else:
            raise TypeError(
                f"Expected Structure or StructureGraph, got {type(structure_or_structuregraph)}"
            )

        return self._featurize(structure_graph)
