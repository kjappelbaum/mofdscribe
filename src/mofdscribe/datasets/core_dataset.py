# -*- coding: utf-8 -*-
"""CoRE Dataset."""
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .checks import check_all_file_exists, length_check
from .dataset import StructureDataset
from ..constants import MOFDSCRIBE_PYSTOW_MODULE


class CoREDataset(StructureDataset):
    """Dataset of gas uptake related features for a subset of CoRE MOFs.

    The labels were computed by Moosavi et al. (2020) [Moosavi2020]_.

    To reduce the risk of data leakage, we  (by default) also only keep one representative
    structure for a "base refcode" (i.e. the first five letters of a refcode).
    For instance, the base refcode for IGAHED001 is IGAHED. Structures with same
    base refcode but different refcodes are often different refinements, or measurements
    at different temperatures and hence chemically quite similar. For instance,
    the base refcode `TOWPEC` would appear 60 times, `NARVUA` 22 times and so on.
    Additionally, we (by default) only keep one structure per "structure hash"
    which is an approximate graph-isomoprhism check, assuming the VESTA bond thresholds
    for the derivation of the structure graph.

    .. warning::
        Even though we performed some basic sanity checks, there are currently still
        some structures that might chemically not be reasonable.
        Also, even though we only keep one structure per base refcode, there is still
        potential for data leakge. We urge users to still drop duplicates (or close neighbors)
        after featurization.

        If this set is used as test set, make sure to drop all overlapping entries in your training set.

    The years refer to the publication dates of the paper crossreferenced
    in the CSD entry of the structure. We excluded structures that are not
    deposited in the CSD.

    The available labels are:

    * 'pure_CO2_kH': Henry coefficient of CO2 obtained by Widom method in mol kg-1 Pa-1
    * 'pure_CO2_widomHOA': Heat of adsorption of CO2 obtained by Widom method in
    * 'pure_methane_kH': Henry coefficient of methane obtained by Widom method in mol kg-1 Pa-1
    * 'pure_methane_widomHOA': Heat of adsorption of methane obtained by Widom method
    * 'pure_uptake_CO2_298.00_15000': Pure CO2 uptake at 298.00 K and 15000 Pa in mol kg-1
    * 'pure_uptake_CO2_298.00_1600000': Pure CO2 uptake at 298.00 K and 1600000 Pa in mol kg-1
    * 'pure_uptake_methane_298.00_580000': Pure methane uptake at 298.00 K and 580000 Pa in mol kg-1
    * 'pure_uptake_methane_298.00_6500000': Pure methane uptake at 298.00 K and 6500000 Pa in mol kg-1
    * 'logKH_CO2': Logarithm of Henry coefficient of CO2 obtained by Widom method
    * 'logKH_CH4': Logarithm of Henry coefficient of methane obtained by Widom method
    * 'CH4DC': CH4 deliverable capacity in vSTP/v
    * 'CH4HPSTP': CH4 high pressure uptake in standard temperature and pressure in vSTP/v
    * 'CH4LPSTP': CH4 low pressure uptake in standard temperature and pressure in vSTP/v

    References
    ----------
    .. [Moosavi2020] Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet, J. P.; Boyd, P. G.; Lee,
        Y.; Smit, B.; Kulik, H. J. Understanding the Diversity of the Metal-Organic Framework Ecosystem.
        Nature Communications 2020, 11 (1), 4068. https://doi.org/10.1038/s41467-020-17755-8.

    """

    _files = {
        'v0.0.1': {
            'df': 'https://www.dropbox.com/s/i2khfte4s6mcg30/data.json?dl=1',
            'structures': 'https://www.dropbox.com/s/1v7zknesttixw68/structures.tar.gz?dl=1',
            'expected_length': 8821,
        }
    }

    def __init__(
        self,
        version: Optional[str] = 'v0.0.1',
        drop_basename_duplicates: Optional[bool] = True,
        drop_graph_duplicates: Optional[bool] = True,
    ):
        """Construct an instance of the CoRE dataset.

        Args:
            version (Optional[str], optional): version number to use.
                Defaults to "v0.0.1".
            drop_basename_duplicates (Optional[bool], optional): If True, keep only one structure
                per CSD basename. Defaults to True.
            drop_graph_duplicates (Optional[bool], optional): If True, keep only one structure
                per decorated graph hash. Defaults to True.

        Raises:
            ValueError: If the provided version number is not available.
        """
        if version not in self._files:
            raise ValueError(
                f'Version {version} not available. Available versions: {list(self._files.keys())}'
            )
        self.version = version

        self._structure_dir = MOFDSCRIBE_PYSTOW_MODULE.ensure_untar(
            'CoRE',
            self.version,
            name='structures.tar.gz',
            url=self._files[version]['structures'],
        )

        self._df = pd.DataFrame(
            MOFDSCRIBE_PYSTOW_MODULE.ensure_json(
                'CoRE', self.version, name='data.json', url=self._files[version]['df']
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
            os.path.join(self._structure_dir, f + '.cif') for f in self._df['name_y']
        ]

        check_all_file_exists(self._structures)

        self._years = self._df['year']
        self._decorated_graph_hashes = self._df['hash']
        self._undecorated_graph_hashes = self._df['undecorated_hash']
        self._decorated_scaffold_hashes = self._df['scaffold_hash']
        self._undecorated_scaffold_hashes = self._df['undecorated_scaffold_hash']
        self._densities = self._df['density_x']
        self._labelnames = (
            'pure_CO2_kH',
            'pure_CO2_widomHOA',
            'pure_methane_kH',
            'pure_methane_widomHOA',
            'pure_uptake_CO2_298.00_15000',
            'pure_uptake_CO2_298.00_1600000',
            'pure_uptake_methane_298.00_580000',
            'pure_uptake_methane_298.00_6500000',
            'logKH_CO2',
            'logKH_CH4',
            'CH4DC',
            'CH4HPSTP',
            'CH4LPSTP',
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
            '@article{Moosavi2020,'
            'doi = {10.1038/s41467-020-17755-8},'
            'url = {https://doi.org/10.1038/s41467-020-17755-8},'
            'year = {2020},'
            'month = aug,'
            'publisher = {Springer Science and Business Media {LLC}},'
            'volume = {11},'
            'number = {1},'
            'author = {Seyed Mohamad Moosavi and Aditya Nandy and Kevin Maik Jablonka '
            'and Daniele Ongari and Jon Paul Janet and Peter G. Boyd and Yongjin Lee '
            'and Berend Smit and Heather J. Kulik},'
            'title = {Understanding the diversity of the metal-organic framework ecosystem},'
            'journal = {Nature Communications}'
            '}',
            '@article{Chung2019,'
            'doi = {10.1021/acs.jced.9b00835},'
            'url = {https://doi.org/10.1021/acs.jced.9b00835},'
            'year = {2019},'
            'month = nov,'
            'publisher = {American Chemical Society ({ACS})},'
            'volume = {64},'
            'number = {12},'
            'pages = {5985--5998},'
            'author = {Yongchul G. Chung and Emmanuel Haldoupis and Benjamin J. Bucior '
            'and Maciej Haranczyk and Seulchan Lee and Hongda Zhang and '
            'Konstantinos D. Vogiatzis and Marija Milisavljevic and Sanliang Ling '
            'and Jeffrey S. Camp and Ben Slater and J. Ilja Siepmann and '
            'David S. Sholl and Randall Q. Snurr},'
            'title = {Advances,  Updates,  and Analytics for the Computation-Ready, '
            'Experimental Metal{\textendash}Organic Framework Database: {CoRE} {MOF} 2019},'
            r'journal = {Journal of Chemical {\&}amp$\mathsemicolon$ Engineering Data}'
            '}',
            '@article{Chung2014,'
            'doi = {10.1021/cm502594j},'
            'url = {https://doi.org/10.1021/cm502594j},'
            'year = {2014},'
            'month = oct,'
            'publisher = {American Chemical Society ({ACS})},'
            'volume = {26},'
            'number = {21},'
            'pages = {6185--6192},'
            'author = {Yongchul G. Chung and Jeffrey Camp and '
            'Maciej Haranczyk and Benjamin J. Sikora and Wojciech Bury '
            'and Vaiva Krungleviciute and Taner Yildirim and Omar K. Farha '
            'and David S. Sholl and Randall Q. Snurr},'
            'title = {Computation-Ready,  Experimental Metal{\textendash}Organic Frameworks: '
            'A Tool To Enable High-Throughput Screening of Nanoporous Crystals},'
            'journal = {Chemistry of Materials}'
            '}',
        ]
