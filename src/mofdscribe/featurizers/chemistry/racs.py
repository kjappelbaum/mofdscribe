# -*- coding: utf-8 -*-
"""Revised autocorrelation functions (RACs) for MOFs."""
from collections import OrderedDict, defaultdict
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from element_coder import encode
from loguru import logger
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IStructure, Structure

from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.aggregators import AGGREGATORS, ARRAY_AGGREGATORS
from mofdscribe.featurizers.utils.extend import operates_on_istructure, operates_on_structure
from mofdscribe.featurizers.utils.structure_graph import (
    get_connected_site_indices,
    get_neighbors_at_distance,
    get_structure_graph,
)

from ._fragment import get_bb_indices

__all__ = ("RACS",)


def _compute_racs(
    start_indices: Iterable[int],
    structure_graph: StructureGraph,
    properties: Tuple[str],
    scope: int,
    property_aggregations: Tuple[str],
    corr_aggregations: Tuple[str],
    part_name: str = "",
    nan_value: float = np.nan,
):
    racs = defaultdict(lambda: defaultdict(list))
    if len(start_indices) == 0:
        logger.debug(f"No start indices for {part_name}")
        for prop in properties:
            for agg in property_aggregations:
                racs[prop][agg].append(nan_value)
    else:
        for start_atom in start_indices:
            _, neighbors = get_neighbors_at_distance(structure_graph, start_atom, scope)
            num_neighbors = len(neighbors)
            # We only branch if there are actually neighbors scope many bonds away ...

            if num_neighbors > 0:
                site = structure_graph.structure[start_atom]

                for neighbor in neighbors:

                    n = structure_graph.structure[neighbor]
                    for prop in properties:
                        if prop in ("I", 1):
                            p0 = 1
                            p1 = 1
                        elif prop == "T":
                            p0 = num_neighbors
                            p1 = len(get_connected_site_indices(structure_graph, neighbor))
                        else:
                            p0 = encode(site.specie.symbol, prop)
                            p1 = encode(n.specie.symbol, prop)

                        for agg in property_aggregations:
                            agg_func = AGGREGATORS[agg]
                            racs[prop][agg].append(agg_func((p0, p1)))
            else:
                logger.debug(f"No neighbors found for {start_atom}")
                for prop in properties:
                    for agg in property_aggregations:
                        racs[prop][agg].append(nan_value)
    aggregated_racs = {}

    for property_name, property_values in racs.items():
        for aggregation_name, aggregation_values in property_values.items():
            for corr_agg in corr_aggregations:
                agg_func = ARRAY_AGGREGATORS[corr_agg]
                name = f"racs_bb-{part_name}_prop-{property_name}_scope-{scope}_propagg-{aggregation_name}_corragg-{corr_agg}"  # noqa: E501
                aggregated_racs[name] = agg_func(aggregation_values)

    return aggregated_racs


def _get_racs_for_bbs(
    bb_indices: Iterable[int],
    structure_graph: StructureGraph,
    properties: Tuple[str],
    scopes: List[int],
    property_aggregations: Tuple[str],
    corr_aggregations: Tuple[str],
    bb_aggregations: Tuple[str],
    bb_name: str = "",
):
    bb_racs = defaultdict(list)

    if not bb_indices:
        # one nested list to make it trigger filling it with nan values
        bb_indices = [[]]
    for start_indices in bb_indices:
        for scope in scopes:
            racs = _compute_racs(
                start_indices,
                structure_graph,
                properties,
                scope,
                property_aggregations,
                corr_aggregations,
                bb_name,
            )
            for k, v in racs.items():
                bb_racs[k].append(v)

    aggregated_racs = {}
    for racs_name, racs_values in bb_racs.items():
        for bb_agg in bb_aggregations:
            agg_func = ARRAY_AGGREGATORS[bb_agg]
            name = f"{racs_name}_bbagg-{bb_agg}"
            aggregated_racs[name] = agg_func(racs_values)

    return aggregated_racs


@operates_on_istructure
@operates_on_structure
class RACS(MOFBaseFeaturizer):
    r"""Modified version of the revised autocorrelation functions (RACs) for MOFs.

    Originally proposed by Moosavi et al. (10.1038/s41467-020-17755-8).
    In the original paper, RACs were computed as

    .. math::
        {}_{{\rm{scope}}}^{{\rm{start}}}{P}_{d}^{{\rm{diff}}}=\mathop{\sum }\limits_{i}^{{\rm{start}}}\mathop{\sum }\limits_{j}^{{\rm{scope}}}({P}_{i}-{P}_{j})\delta ({d}_{i,j},d). # noqa: E501

    Here, we allow to replace the double sum by different aggregations. We call
    this `corr_agg`. The default `sum` is equivalent to the original RACS.
    Moreover, the implementation here keeps track of different linker/node
    molecules and allows to compute and aggregate the RACS for each molecule
    separately. The `bb_agg` feature then determines how those RACs for each BB
    are aggregated. The `sum` is equivalent to the original RACS (i.e. all
    applicable linker atoms would be added to the start/scope lists).

    Note that the "bbs" define the start atoms. However, we still consider the whole
    neighborhood of each start atom (i.e., including atoms that are not part of this "bb").

    Furthermore, here we allow to allow any of the
    `element-coder <https://github.com/kjappelbaum/element-coder>`_ properties to be used as
    `property` :math:`P_{i}`.

    To use to original implementation, see `molSimplify
    <https://github.com/hjkgrp/molSimplify>`_.
    """

    def __init__(
        self,
        attributes: Tuple[Union[int, str]] = ("X", "mod_pettifor", "I", "T"),
        scopes: Tuple[int] = (1, 2, 3),
        prop_agg: Tuple[str] = ("product", "diff"),
        corr_agg: Tuple[str] = ("sum", "avg"),
        bb_agg: Tuple[str] = ("avg", "sum"),
        bond_heuristic: str = "vesta",
        bbs: Optional[Tuple[str]] = (
            "linker_all",
            "linker_connecting",
            "linker_functional",
            "linker_scaffold",
            "nodes",
        ),
        primitive: bool = True,
    ) -> None:
        """
        Initialize the RACS featurizer.

        Args:
            attributes (Tuple[Union[int, str]]): Properties that are correlated.
                Defaults to ("X", "mod_pettifor", "I", "T").
            scopes (Tuple[int]): Number of edges to traverse. Defaults to (1, 2, 3).
            prop_agg (Tuple[str]): Function for aggregating the properties.
                Defaults to ("product", "diff").
            corr_agg (Tuple[str]): Function to aggregate the properties across different start/scopes.
                Defaults to ("sum", "avg").
            bb_agg (Tuple[str]): Function used to aggregate the properties across different building blocks.
                Defaults to ("avg", "sum").
            bond_heuristic (str): Method used to guess bonds. Defaults to "vesta".
            bbs (Tuple[str]): Building blocks to use. Defaults to ("linker_all",
                "linker_connecting", "linker_functional", "linker_scaffold", "nodes").
            primitive (bool): If True, the structure is reduced to its primitive
                form before the descriptor is computed. Defaults to True.
        """
        self.attributes = attributes
        self.scopes = scopes
        self.prop_agg = prop_agg
        self.corr_agg = corr_agg
        self.bb_agg = bb_agg
        self.bond_heuristic = bond_heuristic
        if bbs is not None:
            self._bbs = bbs
        else:
            self._bbs = [
                "linker_all",
                "linker_connecting",
                "linker_functional",
                "linker_scaffold",
                "nodes",
            ]
        super().__init__(primitive=primitive)

    def _featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        if isinstance(structure, Structure):
            structure = IStructure.from_sites(structure)
        sg = get_structure_graph(structure, self.bond_heuristic)

        racs = {}
        # This finds all indices for a particular subset of atoms
        # e.g. "nodes", "linker_all", "linker_connecting", "linker_functional", "linker_scaffold"
        bb_indices = get_bb_indices(sg)
        for bb in self._bbs:
            racs.update(
                _get_racs_for_bbs(
                    bb_indices[bb],
                    sg,
                    self.attributes,
                    self.scopes,
                    self.prop_agg,
                    self.corr_agg,
                    self.bb_agg,
                    bb,
                )
            )
        racs_ordered = OrderedDict(sorted(racs.items()))
        return np.array(list(racs_ordered.values()))

    def _get_feature_labels(self) -> List[str]:
        names = []
        for bb in self._bbs:
            for scope in self.scopes:
                for prop in self.attributes:
                    for property_agg in self.prop_agg:
                        for cor_agg in self.corr_agg:
                            for bb_agg in self.bb_agg:
                                names.append(
                                    f"racs_bb-{bb}_prop-{prop}_scope-{scope}_propagg-{property_agg}_corragg-{cor_agg}_bbagg-{bb_agg}"  # noqa: E501
                                )

        names = sorted(names)
        return names

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def citations(self) -> List[str]:
        return [
            "@article{Moosavi2020,"
            "doi = {10.1038/s41467-020-17755-8},"
            "url = {https://doi.org/10.1038/s41467-020-17755-8},"
            "year = {2020},"
            "month = aug,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {11},"
            "number = {1},"
            "author = {Seyed Mohamad Moosavi and Aditya Nandy and "
            "Kevin Maik Jablonka and Daniele Ongari and Jon Paul Janet "
            "and Peter G. Boyd and Yongjin Lee and Berend Smit and Heather J. Kulik},"
            "title = {Understanding the diversity of the metal-organic framework ecosystem},"
            "journal = {Nature Communications}"
            "}",
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
