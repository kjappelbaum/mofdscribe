# -*- coding: utf-8 -*-
"""MOF class - the container consumed by all featurizers."""
from collections import namedtuple
from typing import Optional

from backports.cached_property import cached_property
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import IStructure, Structure
from structuregraph_helpers.create import get_structure_graph
from structuregraph_helpers.hash import (
    decorated_graph_hash,
    decorated_scaffold_hash,
    undecorated_graph_hash,
    undecorated_scaffold_hash,
)

from mofdscribe.types import PathType, StructureIStructureType


class MOF:
    """A container for a MOF structure.

    The MOF class is the container for a MOF structure. It contains at least
    the structure itself, but can also contain additional information such as

    - the structure graph
    - the fragments
    - the hashes

    The MOF class is a conenient input for the featurizers because
    they can simply take whatever they need from the MOF class.
    Additionally, expensive computations such as the structure graph
    or the fragments can be cached in the MOF class.

    This not only simplifies the featurizer code, but also makes it
    possible to reuse the artifacts of the MOF class for other purposes.

    .. note::

        By default, the MOF class will use :py:obj:`~pymatgen.core.structure.IStructure`
        as the structure container. This is because we want to make sure that there is
        no way tht the inputs are mutated during the featurization process.

        However, some aspects of ``matminer`` require the use of :py:obj:`~pymatgen.core.structure.Structure`
        as the structure container. The corresponding matminer adapter will deal with this.
        However, we cannot strictly enfore immutability of the structure container in this case.
    """

    def __init__(
        self,
        structure: StructureIStructureType,
        structure_graph: Optional[StructureGraph] = None,
        local_env_strategy: str = "vesta",
        fragmentation_kwargs: Optional[dict] = None,
    ):
        """Initialize a MOF class.

        Args:
            structure (StructureIStructureType): The structure of the MOF as
                :py:class:`~pymatgen.core.Structure` or  :py:class:`~pymatgen.core.IStructure`.
            structure_graph (Optional[StructureGraph]): The structure graph of the MOF.
                If not provided, it will be computed on the fly.
            local_env_strategy (str): The local environment strategy to use for the structure graph.
                Defaults to ``"vesta"``.
            fragmentation_kwargs: The fragmentation kwargs to use for the fragmentation using ``moffragmentor``.
                Defaults to ``None``.
                Some relevant kwargs are:
                * check_dimensionality (bool): Check if the node is 0D.
                    If not, split into isolated metals.
                    Defaults to True.
                * create_single_metal_bus (bool): Create a single metal BUs.
                    Defaults to False.
                * break_organic_nodes_at_metal (bool): Break nodes into single metal BU
                    if they appear "too organic".

                For the dimensionality featurizers applied to building blocks, you want
                to be careful wtih ``check_dimensionality``.
        """
        self.__structure = (
            IStructure.from_sites(structure.sites)
            if isinstance(structure, Structure)
            else structure
        )
        self.__structure_graph = structure_graph
        self.__local_env_strategy = local_env_strategy
        self.__fragmentation_kwargs = (
            fragmentation_kwargs if fragmentation_kwargs is not None else {}
        )

    @property
    def structure(self) -> IStructure:
        """Get the structure of the MOF."""
        return self.__structure

    @property
    def structure_graph(self) -> StructureGraph:
        """Get the structure graph of the MOF."""
        if self.__structure_graph is None:
            self.__structure_graph = get_structure_graph(
                self.__structure, self.__local_env_strategy
            )
        return self.__structure_graph

    @classmethod
    def from_file(cls, path: PathType, primitive: bool = True, fragmentation_kwargs: Optional[dict]=None) -> "MOF":
        """Create a MOF class from a file.

        Args:
            path (PathType): The path to the file.
            primitive (bool): Whether to use the primitive cell or not.
                Defaults to ``True``.
            fragmentation_kwargs: The fragmentation kwargs to use for the fragmentation using ``moffragmentor``.

        Returns:
            MOF: The MOF class.
        """
        return cls(
            IStructure.from_file(path).get_primitive_structure()
            if primitive
            else IStructure.from_file(path),
            fragmentation_kwargs=fragmentation_kwargs,
        )

    @cached_property
    def fragments(self) -> namedtuple:
        """
        Get the fragments of the MOF.

        Raises:
            ImportError: If ``moffragmentor`` is not installed.

        Returns:
            namedtuple: :py:class:`~moffragmentor.fragment.Fragments` namedtuple.
        """
        try:
            from moffragmentor import MOF as MOFFRAGMENTORMOF
        except ImportError:
            raise ImportError(
                "moffragmentor is not installed. Please install it to use the fragments feature.\
                See https://github.com/kjappelbaum/moffragmentor for more information."
            )

        mof = MOFFRAGMENTORMOF(self.__structure, self.structure_graph)
        return mof.fragment(**self.__fragmentation_kwargs)

    @cached_property
    def decorated_graph_hash(self) -> str:
        """Return the Weisfeiler-Lehman graph hash.

        Hashes are identical for isomorphic graphs
        (taking the atomic kinds into account)
        and there are guarantees that non-isomorphic graphs will get different hashes.

        Returns:
            str: Graph hash
        """
        return decorated_graph_hash(self.structure_graph, lqg=False)

    @cached_property
    def decorated_scaffold_hash(self) -> str:
        """Return the Weisfeiler-Lehman scaffold hash.

        The scaffold is the graph with the all terminal groups and
        atoms removed (i.e., formally, bridges are broken).

        Returns:
            str: Scaffold hash
        """
        return decorated_scaffold_hash(self.structure_graph, lqg=False)

    @cached_property
    def undecorated_graph_hash(self) -> str:
        """Return the Weisfeiler-Lehman graph hash.

        Hashes are identical for isomorphic graphs
        and there are guarantees that non-isomorphic graphs will get different hashes.
        Undecorated means that the atomic numbers are not taken into account.

        Returns:
            str: Graph hash
        """
        return undecorated_graph_hash(self.structure_graph, lqg=False)

    @cached_property
    def undecorated_scaffold_hash(self) -> str:
        """Return the Weisfeiler-Lehman scaffold hash.

        The scaffold is the graph with the all terminal groups and
        atoms removed (i.e., formally, bridges are broken).
        Undecorated means that the atomic numbers are not taken into account.

        Returns:
            str: Scaffold hash
        """
        return undecorated_scaffold_hash(self.structure_graph, lqg=False)
