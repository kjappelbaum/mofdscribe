# -*- coding: utf-8 -*-
"""RACs on molecule and structure graphs with optional community detection."""
from collections import OrderedDict, defaultdict
from typing import Collection, Dict, List, Optional, Set, Tuple, Union

import networkx.algorithms.community as nx_comm
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.chemistry.racs import compute_racs
from mofdscribe.featurizers.utils.aggregators import ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.definitions import (
    ALL_ELEMENTS,
    ALL_ELEMENTS_EXCEPT_C_H,
    ALL_METAL_ELEMENTS,
    ALL_NONMETAL_ELEMENTS,
)
from mofdscribe.featurizers.utils.extend import (
    operates_on_moleculegraph,
    operates_on_structuregraph,
)
from mofdscribe.featurizers.utils.structure_graph import get_neighbors_up_to_scope
from mofdscribe.mof import MOF


def _get_site_iter(structuregraph: Union[StructureGraph, MoleculeGraph]):
    if isinstance(structuregraph, StructureGraph):
        return structuregraph.structure.sites
    elif isinstance(structuregraph, MoleculeGraph):
        return structuregraph.molecule.sites
    else:
        raise ValueError("structuregraph must be either a StructureGraph or a MoleculeGraph")


def _get_atom_site_indices(
    structuregraph: Union[StructureGraph, MoleculeGraph],
    atom_groups: Collection[Tuple[str, List[str], bool]],
):
    """Get the indices of the sites that belong to the atom groups."""
    atom_grouped_indices = defaultdict(set)
    for atom_group_name, elements, _no_terminal in atom_groups:
        for i, site in enumerate(_get_site_iter(structuregraph)):
            if site.specie.symbol in elements:
                atom_grouped_indices[atom_group_name].add(i)
    return atom_grouped_indices


def _split_up_communities(
    structuregraph: Union[StructureGraph, MoleculeGraph],
    communities: List[List[int]],
    atom_groups: Collection[Tuple[str, List[str], bool]],
):
    """Create dictionary of communities with atom groups as keys."""
    atom_grouped_communities = defaultdict(list)
    indices_to_atom_group = _get_atom_site_indices(
        structuregraph=structuregraph, atom_groups=atom_groups
    )

    for atom_group_name, _elements, _no_terminal in atom_groups:
        for community in communities:
            if not isinstance(community, set):
                communities_set = set([community])
            else:
                communities_set = community
            atom_grouped_communities[atom_group_name].append(
                indices_to_atom_group[atom_group_name] & set(communities_set)
            )

    return atom_grouped_communities


# ToDo: generalize the code and implement in only one place
# the main racs code is quite similar to this one
def _get_racs_for_community(
    community_indices,
    structure_graph: StructureGraph,
    neighbors_at_distance: Dict[int, Dict[int, Set[int]]],
    properties: Tuple[str],
    scopes: List[int],
    property_aggregations: Tuple[str],
    corr_aggregations: Tuple[str],
    community_aggregations: Tuple[str],
    community_name: str = "",
):
    community_racs = defaultdict(list)

    if not community_indices:
        # one nested list to make it trigger filling it with nan values
        community_indices = [[]]
    for start_indices in community_indices:
        for scope in scopes:
            racs = compute_racs(
                start_indices,
                structure_graph,
                neighbors_at_distance,
                properties,
                scope,
                property_aggregations,
                corr_aggregations,
                community_name,
            )
            for k, v in racs.items():
                community_racs[k].append(v)

    aggregated_racs = {}
    for racs_name, racs_values in community_racs.items():
        for community_agg in community_aggregations:
            agg_func = ARRAY_AGGREGATORS[community_agg]
            name = f"{racs_name}__atomgroupagg-{community_agg}"
            aggregated_racs[name] = agg_func(racs_values)

    return aggregated_racs


@operates_on_structuregraph
@operates_on_moleculegraph
class ModularityCommunityCenteredRACS(MOFBaseFeaturizer):
    """RACs on molecule and structure graphs with optional community detection.

    This featurizer is a flavor of the :ref:`RACs <RACs>` featurizer.
    It can split the computation over user-defined atom groups and automatically determined communities.
    For determining communities, we use ``networkx``'s implementation
    of greedy modularity maximization [NewmanModularity]_.

    The features are computed for each communinty within each atom group and then aggregated over the communities.
    That is, the number of features will depend on the number of atom groups.
    The communities will only impact how the features within an atom group are aggregated.

    There are different ways in which you might want to use this featurizer.

    1) No prior on communitiy structure. In this case, you can set ``dont_use_communities=True``.
    It will still compute the RACs over the atom groups you specify but won't have an inner loop over communities
    that are then aggegated for the final features for a given atom group.

    2) No prior on atom grouping. In this case, you can set ``atom_groups=None``.
    This will compute the RACs over all atoms.
    """

    _NAME = "ModularityCommunityCenteredRACS"

    def __init__(
        self,
        atom_groups: Optional[Collection[Tuple[str, Collection[str], bool]]] = None,
        attributes: Tuple[Union[int, str]] = ("X", "mod_pettifor", "I", "T"),
        scopes: Tuple[int] = (1, 2, 3),
        prop_agg: Tuple[str] = ("product", "diff"),
        corr_agg: Tuple[str] = ("sum", "avg"),
        atom_groups_agg: Tuple[str] = ("avg", "sum"),
        dont_use_communities: bool = False,
    ):
        """Constuct a ModularityCommunityCenteredRACS featurizer.

        Features are computed for each atom group and for each community. The features are then aggregated
        over the communities.

        Args:
            atom_groups (Optional[Collection[Tuple[str, Collection[str], bool]]], optional):
                Elements which form start scopes for RACs.
                The first element is the name of the atom group, and will appear in the feature name.
                The second element is a list of elements that belong to the atom group. For example,
                ('C-H', ['C', 'H'], False) will create a feature for all carbon atoms and hydrogen atoms.
                The third element of the tuple is currently not used (but will be used as a flag to
                indicate whether to include terminal/leading to terminal atoms in the start scope).
                If set to None, there will be one atom group with all atoms.
                Defaults to None.
            attributes (Tuple[Union[int, str]], optional):
                Elemental properties used for the construction of RACs. Defaults to ("X", "mod_pettifor", "I", "T").
            scopes (Tuple[int], optional): Scopes used for the construction of RACs.
                Defaults to (1, 2, 3).
            prop_agg (Tuple[str], optional): Aggregation methods used for the aggregation of
                properties. "Product" corresponds to "product-RACs" and "diff" to "difference-RACs".
                Defaults to ("product", "diff").
            corr_agg (Tuple[str], optional): Aggregation methods used for the aggregation of the
                correlated properties. Defaults to ("sum", "avg").
            atom_groups_agg (Tuple[str], optional): Aggregation methods used for the pooling
                over atom communities within one atom group. Defaults to ("avg", "sum").
            dont_use_communities (bool): If set to true, we do not use modularity-based community detection.
                Features are then simply averaged over all atoms.
                Defaults to False.
        """
        if atom_groups is None:
            atom_groups = [("all", ALL_ELEMENTS, False)]
        self.atom_groups = atom_groups
        self.attributes = attributes
        self.scopes = scopes
        self.prop_agg = prop_agg
        self.corr_agg = corr_agg
        self.atom_groups_agg = atom_groups_agg
        self.dont_use_communities = dont_use_communities

    @classmethod
    def from_preset(cls, preset: str, **kwargs):
        if preset.lower() == "cof":
            atom_groups = [
                ("all", ALL_ELEMENTS, False),
                ("C-H", {"C", "H"}, False),
                ("not_C-H", ALL_ELEMENTS_EXCEPT_C_H, False),
            ]
        elif preset.lower() == "mof":
            atom_groups = [
                ("all", ALL_ELEMENTS, False),
                ("metal", ALL_METAL_ELEMENTS, False),
                ("nonmetal", ALL_NONMETAL_ELEMENTS, False),
            ]
        return cls(atom_groups=atom_groups, **kwargs)

    def _featurize(self, structuregraph: Union[StructureGraph, MoleculeGraph]):
        if not self.dont_use_communities:
            communities = list(
                nx_comm.greedy_modularity_communities(structuregraph.graph.to_undirected())
            )
        else:
            communities = [range(len(structuregraph))]  # make one community with all atoms

        neighbors_at_distance = {
            i: get_neighbors_up_to_scope(structuregraph, i, max(self.scopes))
            for i in range(len(structuregraph))
        }

        groups = _split_up_communities(structuregraph, communities, self.atom_groups)

        racs = {}
        for group_name, group_indices in groups.items():
            racs.update(
                _get_racs_for_community(
                    group_indices,
                    structuregraph,
                    neighbors_at_distance,
                    self.attributes,
                    self.scopes,
                    self.prop_agg,
                    self.corr_agg,
                    self.atom_groups_agg,
                    group_name,
                )
            )

        racs_ordered = OrderedDict(sorted(racs.items()))
        return np.array(list(racs_ordered.values()))

    def featurize(self, mof: MOF):
        return self._featurize(mof.structure_graph)

    def _get_feature_labels(self):
        names = []
        for atom_group_name, _, _ in self.atom_groups:
            for scope in self.scopes:
                for prop in self.attributes:
                    for property_agg in self.prop_agg:
                        for cor_agg in self.corr_agg:
                            for atom_group_agg in self.atom_groups_agg:
                                names.append(
                                    f"{self._NAME}-{atom_group_name}_prop-{prop}_scope-{scope}_propagg-{property_agg}_corragg-{cor_agg}_atomgroupagg-{atom_group_agg}"  # noqa: E501
                                )

        names = sorted(names)
        return names

    def feature_labels(self):
        return self._get_feature_labels()

    def citations(self) -> List[str]:
        return [
            "@article{Janet2017,"
            "doi = {10.1021/acs.jpca.7b08750},"
            "url = {https://doi.org/10.1021/acs.jpca.7b08750},"
            "year = {2017},"
            "month = nov,"
            "publisher = {American Chemical Society ({ACS})},"
            "volume = {121},"
            "number = {46},"
            "pages = {8939--8954},"
            "author = {Jon Paul Janet and Heather J. Kulik},"
            "title = {Resolving Transition Metal Chemical Space: "
            "Feature Selection for Machine Learning and Structure{\textendash}Property Relationships},"
            "journal = {The Journal of Physical Chemistry A}"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
