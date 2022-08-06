import os
import subprocess
import time
from glob import glob
from pathlib import Path

import click

SLURM_SUBMISSION_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB

conda activate mofdscribe
python compute_hashes.py {indir} {outdir} --start {start} --end {end}
"""


SLURM_SUBMISSION_TEMPLATE_DAINT = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=60GB

conda activate mofdscribe
python compute_hashes.py {indir} {outdir} --start {start} --end {end}
"""


@click.command("cli")
@click.argument("indir", type=click.Path(exists=True))
@click.argument("outdir")
@click.option("--daint", is_flag=True, help="Use daint")
def cli(indir, outdir, daint):
    ALL_STRUCTURES = glob(os.path.join(indir, "*.cif"))
    CHUNK_SIZE = 1000
    name = Path(indir).stem
    for start in range(0, len(ALL_STRUCTURES), CHUNK_SIZE):
        end = start + CHUNK_SIZE
        job_name = f"hasher_{name}_{start}_{end}"

        with open(f"{job_name}.slurm", "w") as f:
            if daint:
                f.write(
                    SLURM_SUBMISSION_TEMPLATE_DAINT.format(
                        indir=indir, outdir=outdir, start=start, end=end, job_name=job_name
                    )
                )
                subprocess.run(["sbatch", "-A", "pr128", "-C", "gpu", f"{job_name}.slurm"])
                time.sleep(0.5)
            else:
                f.write(
                    SLURM_SUBMISSION_TEMPLATE.format(
                        indir=indir, outdir=outdir, start=start, end=end, job_name=job_name
                    )
                )
                subprocess.run(["sbatch", f"{job_name}.slurm"])
                time.sleep(0.5)


if __name__ == "__main__":
    cli()
