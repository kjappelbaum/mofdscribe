# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import List, Tuple
from pervect import PersistenceVectorizer
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import Structure, IStructure

# ToDo: implement https://github.com/scikit-tda/pervect


class PHVect(BaseFeaturizer):
    """

    Computes persistent homology barcodes.
    Typically, persistent barcodes are computed for all atoms in the structure.
    However, one can also compute persistent barcodes for a subset of atoms.
    This can be done by specifying the atom types in the constructor.
    """

    def __init__(
        self,
        atom_types=Tuple[str],
        n_components: int = 20,
        apply_umap: bool = False,
        umap_n_components: int = 2,
        umap_metric: str = "hellinger",
        p: int = 1,
        random_state=None,
    ) -> None:
        self.transformers = defaultdict(lambda: defaultdict(dict))
        self.n_components = n_components
        self.apply_umap = apply_umap
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.p = p
        self.random_state = random_state

    def feature_labels(self) -> List[str]:
        return ...

    def fit(self, structures: List[Structure, IStructure]):
        return self

    def citations(self):
        return [
            "@article{perea2019approximating,"
            "title   = {Approximating Continuous Functions on Persistence Diagrams Using Template Functions},"
            "author  = {Jose A. Perea and Elizabeth Munch and Firas A. Khasawneh},"
            "year    = {2019},"
            "journal = {arXiv preprint arXiv: Arxiv-1902.07190}"
            "}",
            "@article{tymochko2019adaptive,"
            "title   = {Adaptive Partitioning for Template Functions on Persistence Diagrams},"
            "author  = {Sarah Tymochko and Elizabeth Munch and Firas A. Khasawneh},"
            "year    = {2019},"
            "journal = {arXiv preprint arXiv: Arxiv-1910.08506}"
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
