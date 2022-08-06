import os
from glob import glob
from pathlib import Path

import click
from structuregraph_helpers.cli import create_hashes_for_structure
from structuregraph_helpers.utils import dump_json


def compute_hashes(infile, outdir):
    """Compute and dump hashes for infile"""
    hashes = create_hashes_for_structure(infile)
    name = Path(infile).stem
    dump_json(hashes, os.path.join(outdir, f"{name}.json"))


@click.command("cli")
@click.argument("infile", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
@click.option("--start", type=int, default=0)
@click.option("--end", type=int, default=-1)
def main(infile, outdir, start, end):
    """Compute hashes for all CIF files in infile"""
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cif_files = glob(os.path.join(infile, "*.cif"))
    files = sorted(cif_files)

    files = files[start:end]

    for file in files:
        compute_hashes(file, outdir)


if __name__ == "__main__":
    main()
