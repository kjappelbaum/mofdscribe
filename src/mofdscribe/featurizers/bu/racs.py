# -*- coding: utf-8 -*-
from typing import Tuple, Union, List, Optional, Collection, Dict, Set
import networkx.algorithms.community as nx_comm
from mofdscribe.featurizers.base import MOFBaseFeaturizer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from mofdscribe.utils.extend import operates_on_moleculegraph, operates_on_structuregraph
from mofdscribe.mof import MOF
from element_coder.data.coding_data import get_coding_dict
from mofdscribe.utils.structuregraph import get_neighbors_up_to_scope
from collections import defaultdict
from mofdscribe.featurizers.utils.aggregators import AGGREGATORS, ARRAY_AGGREGATORS
from mofdscribe.featurizers.chemistry.racs import compute_racs
from collections import OrderedDict

def _get_site_iter(structuregraph: Union[StructureGraph, MoleculeGraph]):
    if isinstance(structuregraph, StructureGraph):
        return structuregraph.structure.sites
    elif isinstance(structuregraph, MoleculeGraph):
        return structuregraph.molecule.sites
    else:
        raise ValueError("structuregraph must be either a StructureGraph or a MoleculeGraph")

def _get_atom_site_indices(structuregraph: Union[StructureGraph, MoleculeGraph], atom_groups: Collection[Tuple[str, List[str], bool]]):
    """Get the indices of the sites that belong to the atom groups."""
    atom_grouped_indices = defaultdict(set)
    for atom_group_name, elements, _no_terminal in atom_groups:
        for site in _get_site_iter(structuregraph):
            if site.specie.symbol in elements:
                atom_grouped_indices[atom_group_name].add(site.index)
    return atom_grouped_indices

def _split_up_communities(
    structuregraph: Union[StructureGraph, MoleculeGraph],
    communities: List[List[int]],
    atom_groups: Collection[Tuple[str, List[str], bool]],
):
    """Create dictionary of communities with atom groups as keys."""
    atom_grouped_communities = defaultdict(set)
    indices_to_atom_group = _get_atom_site_indices()

    for atom_group_name, elements, _no_terminal in atom_groups:
        for communities in communities:
            atom_grouped_communities[atom_group_name].update(indices_to_atom_group & set(communities))

    return atom_grouped_communities
    
    


_ALL_ELEMENTS = set(get_coding_dict("atomic").keys())
_ALL_ELEMENTS_EXCEPT_C_H = _ALL_ELEMENTS - {"C", "H"}


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
    _NAME = "ModularityCommunityCenteredRACS"

    def __init__(
        self,
        atom_groups: Optional[Collection[Tuple[str, Collection[str], bool]]] = None,
        attributes: Tuple[Union[int, str]] = ("X", "mod_pettifor", "I", "T"),
        scopes: Tuple[int] = (1, 2, 3),
        prop_agg: Tuple[str] = ("product", "diff"),
        corr_agg: Tuple[str] = ("sum", "avg"),
        atom_groups_agg: Tuple[str] = ("avg", "sum"),
        bond_heuristic: str = "vesta",
        dont_use_communities: bool = False,
    ):
        # heteroatom groups is a list of atoms which we will aggregate seperately if they are in a community
        # additional, we can specify if we only aggregate them seperately if they are not terminal or part of a bridge
        # if the atom groups are specified, there should maybe (?) also be an option to specify "all"
        if atom_groups is None:
            atom_groups = [("all", _ALL_ELEMENTS, False)]
        self.atom_groups = atom_groups
        self.attributes = attributes
        self.scopes = scopes
        self.prop_agg = prop_agg
        self.corr_agg = corr_agg
        self.atom_groups_agg = atom_groups_agg
        self.bond_heuristic = bond_heuristic
        self.dont_use_communities = dont_use_communities

    def _featurize(self, structuregraph: Union[StructureGraph, MoleculeGraph]):
        
        if not self.dont_use_communities:
            communities = list(
                nx_comm.greedy_modularity_communities(structuregraph.graph.to_undirected())
            )
        else:
            communities = [range(len(structuregraph))] # make one community with all atoms

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
