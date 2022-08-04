# -*- coding: utf-8 -*-
import concurrent.futures
import os
from glob import glob
from pathlib import Path

import click
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.core import Structure

from mofdscribe.featurizers.chemistry.amd import AMD
from mofdscribe.featurizers.chemistry.aprdf import APRDF
from mofdscribe.featurizers.chemistry.racs import RACS
from mofdscribe.featurizers.pore import PoreDiameters, SurfaceArea
from mofdscribe.featurizers.topology.ph_hist import PHHist
from mofdscribe.featurizers.topology.ph_stats import PHStats

featurizer = MultipleFeaturizer(
    [RACS(), PHStats(), PHHist(), PoreDiameters(), APRDF(), SurfaceArea('CO2'), AMD()]
)


def featurize_mof_ignore_failure(infile, outdir):
    try:
        structure = Structure.from_file(infile)
        feats = featurizer.featurize(structure)
        features = featurizer.feature_labels()
        df = pd.DataFrame([dict(zip(features, feats))])
        inname = Path(infile).stem
        df.to_csv(os.path.join(outdir, inname + '.csv'))

    except Exception as e:
        print(f'Failed to featurize {infile} because of {e}')


def featurize_multiple_mofs(indir, outdir, start, end):
    all_mof_folders = sorted(os.listdir(indir))
    already_featurized = glob(os.path.join(outdir, 'features_*.csv'))
    already_featurized = [Path(f).name.split('_')[-1].split('.')[0] for f in already_featurized]
    mof_folders = all_mof_folders[start:end]
    mof_folders = [f for f in mof_folders if f not in already_featurized]

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for moffolder in mof_folders:
            executor.submit(featurize_mof_ignore_failure, os.path.join(indir, moffolder), outdir)


@click.command('cli')
@click.option('--indir', '-m', default=None, help='Input directory')
@click.option('--outdir', type=click.Path(exists=False), default='.')
@click.option('--start', type=int, default=0)
@click.option('--end', type=int, default=-1)
def cli(indir, outdir, start, end):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    featurize_multiple_mofs(indir, outdir, start, end)


if __name__ == '__main__':
    cli()
