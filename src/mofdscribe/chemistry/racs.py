from typing import Tuple, Union

import networkx as nx
from matminer.featurizers.base import BaseFeaturizer


def correlate_properties(pairs, aggregtations, properties):
    ...


def get_pairs(structure_graph):
    # we could use nx.descendants_at_distance
    ...


class RACS(BaseFeaturizer):
    def __init__(
        self,
        attributes: Tuple[Union[int, str]] = ("X", "electron_affinity"),
        scopes: Tuple[int] = (1, 2, 3),
        prop_agg: Tuple[str] = ("product", "diff"),
        corr_agg: Tuple[str] = ("sum"),
        racs_agg: Tuple[str] = ("sum"),
    ) -> None:
        """Modified version of the RACS for MOFs proposed by Moosavi et al.
        In the original paper, RACs were computed as

        .. math::
            {}_{{\rm{scope}}}^{{\rm{start}}}{P}_{d}^{{\rm{diff}}}=\mathop{\sum }\limits_{i}^{{\rm{start}}}\mathop{\sum }\limits_{j}^{{\rm{scope}}}({P}_{i}-{P}_{j})\delta ({d}_{i,j},d).

        Here, we allow to replace the double sum by different aggregations. We call this `corr_agg`. The default `sum` is equivalent to the original RACS. Moreover, the implementation here keeps track of different linker/node molecules and allows to compute and aggregate the RACS for each molecule separately. The default `sum` is equivalent to the original RACS (i.e. all applicable linker atoms would be added to the start/scope lists).

        To use to original implementation, see `molSimplify <https://github.com/hjkgrp/molSimplify>`_.

        Args:
            attributes (Tuple[Union[int, str]], optional): _description_. Defaults to ("X", "electron_affinity").
            scopes (Tuple[int], optional): _description_. Defaults to (1, 2, 3).
            prop_agg (Tuple[str], optional): _description_. Defaults to ("avg", "product", "diff").
            corr_agg (Tuple[str], optional): _description_. Defaults to ("avg", "product", "diff").
            racs_agg (Tuple[str], optional): _description_. Defaults to ("avg", "product", "diff").
        """
        ...

    def featurize(self, structure):
        ...

    def feature_labels(self):
        ...

    def citations(self):
        return [
            "@article{Moosavi2020,"
            "doi = {10.1038/s41467-020-17755-8},"
            "url = {https://doi.org/10.1038/s41467-020-17755-8},"
            "year = {2020},"
            "month = aug,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {11},"
            "number = {1},"
            "author = {Seyed Mohamad Moosavi and Aditya Nandy and Kevin Maik Jablonka and Daniele Ongari and Jon Paul Janet and Peter G. Boyd and Yongjin Lee and Berend Smit and Heather J. Kulik},"
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
            "title = {Resolving Transition Metal Chemical Space: Feature Selection for Machine Learning and Structure{\textendash}Property Relationships},"
            "journal = {The Journal of Physical Chemistry A}"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
