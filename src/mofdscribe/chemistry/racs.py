# -*- coding: utf-8 -*-
"""Revised autocorrelation functions (RACs) for MOFs"""
from collections import defaultdict
from typing import Iterable, List, Tuple, Union

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IStructure, Structure

from mofdscribe.utils.aggregators import AGGREGATORS, ARRAY_AGGREGATORS
from mofdscribe.utils.structure_graph import (
    get_connected_site_indices,
    get_neighbors_at_distance,
    get_structure_graph,
)

from ._fragment import get_bb_indices

__all__ = 'RACS'


def _compute_racs(
    start_indices: Iterable[int],
    structure_graph: StructureGraph,
    properties: Tuple[str],
    scope: int,
    property_aggregations: Tuple[str],
    corr_aggregations: Tuple[str],
    part_name: str = '',
    nan_value: float = np.nan,
):
    racs = defaultdict(lambda: defaultdict(list))
    if len(start_indices) == 0:
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
                        if prop in ('I', 1):
                            p0 = 1
                            p1 = 1
                        elif prop == 'T':
                            p0 = num_neighbors
                            p1 = len(get_connected_site_indices(structure_graph, neighbor))
                        else:
                            p0 = getattr(site.specie, prop)
                            p1 = getattr(n.specie, prop)

                        for agg in property_aggregations:
                            agg_func = AGGREGATORS[agg]
                            racs[prop][agg].append(agg_func((p0, p1)))
            else:
                for prop in properties:
                    for agg in property_aggregations:
                        racs[prop][agg].append(nan_value)
    aggregated_racs = {}

    for property_name, property_values in racs.items():
        for aggregation_name, aggregation_values in property_values.items():
            for corr_agg in corr_aggregations:
                agg_func = ARRAY_AGGREGATORS[corr_agg]
                name = f'racs_{part_name}_{property_name}_{scope}_{aggregation_name}_{corr_agg}'
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
    bb_name: str = '',
):
    bb_racs = defaultdict(list)

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
            name = f'{racs_name}_{bb_agg}'
            aggregated_racs[name] = agg_func(racs_values)

    return aggregated_racs


class RACS(BaseFeaturizer):
    r"""Modified version of the revised autocorrelation functions (RACs) for
    MOFs proposed by Moosavi et al. (10.1038/s41467-020-17755-8)
    In the original paper, RACs were computed as

    .. math::
        {}_{{\rm{scope}}}^{{\rm{start}}}{P}_{d}^{{\rm{diff}}}=\mathop{\sum }\limits_{i}^{{\rm{start}}}\mathop{\sum }\limits_{j}^{{\rm{scope}}}({P}_{i}-{P}_{j})\delta ({d}_{i,j},d).

    Here, we allow to replace the double sum by different aggregations. We call
    this `corr_agg`. The default `sum` is equivalent to the original RACS.
    Moreover, the implementation here keeps track of different linker/node
    molecules and allows to compute and aggregate the RACS for each molecule
    separately. The `bb_agg` feature then determines how those RACs for each BB
    are aggregated. The `sum` is equivalent to the original RACS (i.e. all
    applicable linker atoms would be added to the start/scope lists).

    Furthermore, here we allow to allow any of the
    :py:class:`pymatgen.core.periodic_table.Specie` properties to be used as
    `property` :math:`P_{i}`.

    To use to original implementation, see `molSimplify
    <https://github.com/hjkgrp/molSimplify>`_.
    """

    def __init__(
        self,
        attributes: Tuple[Union[int, str]] = ('X', 'electron_affinity', 'I', 'T'),
        scopes: Tuple[int] = (1, 2, 3),
        prop_agg: Tuple[str] = ('product', 'diff'),
        corr_agg: Tuple[str] = ('sum',),
        bb_agg: Tuple[str] = ('avg',),
        bond_heuristic: str = 'jmolnn',
    ) -> None:
        """

        Args:
            attributes (Tuple[Union[int, str]], optional): _description_. Defaults to ("X", "electron_affinity", "I", "T").
            scopes (Tuple[int], optional): _description_. Defaults to (1, 2, 3).
            prop_agg (Tuple[str], optional): _description_. Defaults to ("product", "diff").
            corr_agg (Tuple[str], optional): _description_. Defaults to ("sum").
            bb_agg (Tuple[str], optional): _description_. Defaults to ("avg").
            bond_heuristic (str, optional): _description_. Defaults to "jmolnn".
        """
        self.attributes = attributes
        self.scopes = scopes
        self.prop_agg = prop_agg
        self.corr_agg = corr_agg
        self.bb_agg = bb_agg
        self.bond_heuristic = bond_heuristic
        self._bbs = [
            'linker_all',
            'linker_connecting',
            'linker_functional',
            'linker_scaffold',
            'nodes',
        ]

    def featurize(self, structure: Union[Structure, IStructure]) -> np.ndarray:
        if isinstance(structure, Structure):
            structure = IStructure.from_sites(structure)
        sg = get_structure_graph(structure, self.bond_heuristic)
        racs = {}
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

        return np.array(list(racs.values()))

    def _get_feature_labels(self) -> List[str]:
        names = []
        for bb in self._bbs:
            for bb_agg in self.bb_agg:
                for scope in self.scopes:
                    for cor_agg in self.corr_agg:
                        for prop in self.attributes:
                            for property_agg in self.prop_agg:
                                names.append(
                                    f'racs_{bb}_{prop}_{scope}_{property_agg}_{cor_agg}_{bb_agg}'
                                )
        return names

    def feature_labels(self) -> List[str]:
        return self._get_feature_labels()

    def citations(self) -> List[str]:
        return [
            '@article{Moosavi2020,'
            'doi = {10.1038/s41467-020-17755-8},'
            'url = {https://doi.org/10.1038/s41467-020-17755-8},'
            'year = {2020},'
            'month = aug,'
            'publisher = {Springer Science and Business Media {LLC}},'
            'volume = {11},'
            'number = {1},'
            'author = {Seyed Mohamad Moosavi and Aditya Nandy and Kevin Maik Jablonka and Daniele Ongari and Jon Paul Janet and Peter G. Boyd and Yongjin Lee and Berend Smit and Heather J. Kulik},'
            'title = {Understanding the diversity of the metal-organic framework ecosystem},'
            'journal = {Nature Communications}'
            '}',
            '@article{Janet2017,'
            'doi = {10.1021/acs.jpca.7b08750},'
            'url = {https://doi.org/10.1021/acs.jpca.7b08750},'
            'year = {2017},'
            'month = nov,'
            'publisher = {American Chemical Society ({ACS})},'
            'volume = {121},'
            'number = {46},'
            'pages = {8939--8954},'
            'author = {Jon Paul Janet and Heather J. Kulik},'
            'title = {Resolving Transition Metal Chemical Space: Feature Selection for Machine Learning and Structure{\textendash}Property Relationships},'
            'journal = {The Journal of Physical Chemistry A}'
            '}',
        ]

    def implementors(self):
        return ['Kevin Maik Jablonka']
