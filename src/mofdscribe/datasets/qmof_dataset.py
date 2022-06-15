# -*- coding: utf-8 -*-
"""Subsert of the QMOF dataset."""
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .checks import check_all_file_exists, length_check
from .dataset import StructureDataset
from ..constants import MOFDSCRIBE_PYSTOW_MODULE


class QMOFDataset(StructureDataset):
    _files = {
        'v0.0.1': {
            'df': 'https://www.dropbox.com/s/3hls6g6it2agy7u/data.json?dl=1',
            'structures': 'https://www.dropbox.com/s/5k48t12qhlf1hwy/structures.tar.gz?dl=1',
            'expected_length': 15844,
        }
    }

    def __init__(
        self,
        version: Optional[str] = 'v0.0.1',
        drop_basename_duplicates: Optional[bool] = True,
        drop_graph_duplicates: Optional[bool] = True,
    ):

        if version not in self._files:
            raise ValueError(
                f'Version {version} not available. Available versions: {list(self._files.keys())}'
            )
        self.version = version

        self._structure_dir = MOFDSCRIBE_PYSTOW_MODULE.ensure_untar(
            'QMOF',
            self.version,
            name='structures.tar.gz',
            url=self._files[version]['structures'],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                'QMOF', self.version, name='data.json', url=self._files[version]['df']
            )
        ).reset_index(drop=True)

        length_check(self._df, self._files[version]['expected_length'])

        if drop_basename_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=['basename'])
            logger.debug(
                f'Dropped {old_len - len(self._df)} duplicate basenames. New length {len(self._df)}'
            )
        if drop_graph_duplicates:
            old_len = len(self._df)
            self._df = self._df.drop_duplicates(subset=['hash'])
            logger.debug(
                f'Dropped {old_len - len(self._df)} duplicate graphs. New length {len(self._df)}'
            )
        self._df = self._df.reset_index(drop=True)
        self._structures = [
            os.path.join(self._structure_dir, f + '.cif') for f in self._df['qmof_id']
        ]

        check_all_file_exists(self._structures)

        self._years = self._df['year']
        self._decorated_graph_hashes = self._df['hash']
        self._undecorated_graph_hashes = self._df['undecorated_hash']
        self._decorated_scaffold_hashes = self._df['scaffold_hash']
        self._undecorated_scaffold_hashes = self._df['undecorated_scaffold_hash']
        self._densities = self._df['density_x']
        self._labelnames = (
            'outputs.pbe.bandgap',
            'outputs.pbe.cbm',
            'outputs.pbe.vbm',
            'outputs.pbe.directgap',
        )

    @property
    def available_labels(self) -> Tuple[str]:
        return self._labelnames

    def get_labels(
        self, idx: Iterable[int], labelnames: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        labelnames = labelnames if labelnames is not None else self._labelnames
        return self._df.iloc[idx][list(labelnames)].values

    @property
    def citations(self) -> Tuple[str]:
        return [
            '@article{Rosen2021_a,'
            'doi = {10.1016/j.matt.2021.02.015},'
            'url = {https://doi.org/10.1016/j.matt.2021.02.015},'
            'year = {2021},'
            'month = may,'
            'publisher = {Elsevier {BV}},'
            'volume = {4},'
            'number = {5},'
            'pages = {1578--1597},'
            "author = {Andrew S. Rosen and Shaelyn M. Iyer and Debmalya Ray and Zhenpeng Yao and Al{'{a}}n Aspuru-Guzik and Laura Gagliardi and Justin M. Notestein and Randall Q. Snurr},"
            'title = {Machine learning the quantum-chemical properties of metal{\textendash}organic frameworks for accelerated materials discovery},'
            'journal = {Matter}'
            '}',
            '@article{Rosen2021_b,'
            'doi = {10.26434/chemrxiv-2021-6cs91},'
            'url = {https://doi.org/10.26434/chemrxiv-2021-6cs91},'
            'year = {2021},'
            'month = dec,'
            'publisher = {American Chemical Society ({ACS})},'
            'author = {Andrew S. Rosen and Victor Fung and Patrick Huck and Cody T. O{\textquotesingle}Donnell and Matthew K. Horton and Donald G. Truhlar and Kristin A. Persson and Justin M. Notestein and Randall Q. Snurr},'
            'title = {High-Throughput Predictions of Metal{\textendash}Organic Framework Electronic Properties: Theoretical Challenges,  Graph Neural Networks,  and Data Exploration}'
            '}',
        ]
