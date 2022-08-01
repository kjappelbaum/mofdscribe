from glob import glob
import os
import pickle

import pandas as pd
import click
from pathlib import Path
from mofdscribe.featurizers.chemistry.racs import RACS
from mofdscribe.featurizers.chemistry.aprdf import APRDF
from mofdscribe.featurizers.chemistry.amd import AMD
from mofdscribe.featurizers.topology.ph_stats import PHStats
from mofdscribe.featurizers.pore import PoreDiameters, SurfaceArea
from matminer.featurizers.base import MultipleFeaturizer
import concurrent.futures


featurizer = MultipleFeaturizer(
    [RACS(), PHStats(), PoreDiameters("CO2"), APRDF(), SurfaceArea(), AMD()]
)


def featurize_mof_ignore_failure(infile, outdir):
    try:
        feats = featurizer.featurize(infile)
        features = featurizer.feature_labels()
        df = pd.DateFrame(zip(dict(features, feats)))
        inname = Path(infile).stem
        df.to_csv(os.path.join(outdir, inname + ".csv"))

    except Exception as e:
        print(f"Failed to featurize {infile} because of {e}")


def featurize_multiple_mofs(indir, outdir, start, end):
    all_mof_folders = sorted(os.listdir(indir))
    already_featurized = glob(os.path.join(outdir, "features_*.csv"))
    already_featurized = [Path(f).name.split("_")[-1].split(".")[0] for f in already_featurized]
    mof_folders = all_mof_folders[start:end]
    mof_folders = [f for f in mof_folders if f not in already_featurized]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for moffolder in mof_folders:
            executor.submit(featurize_mof_ignore_failure, os.path.join(indir, moffolder), outdir)


@click.command("cli")
@click.option("--indir", "-m", default=None, help="Input directory")
@click.option("--outdir", type=click.Path(exists=False), default=".")
@click.option("--start", type=int, default=0)
@click.option("--end", type=int, default=-1)
def cli(indir, outdir, start, end):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    featurize_multiple_mofs(indir, outdir, start, end)


if __name__ == "__main__":
    cli()
