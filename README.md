
<p align="center">
  <img src="https://github.com/kjappelbaum/mofdscribe/raw/main/docs/source/figures/logo.png" height="300">
</p>
<p align="center">
    <a href="https://github.com/kjappelbaum/mofdscribe/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/kjappelbaum/mofdscribe/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/mofdscribe">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/mofdscribe" />
    </a>
    <a href="https://pypi.org/project/mofdscribe">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/mofdscribe" />
    </a>
    <a href="https://github.com/kjappelbaum/mofdscribe/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/mofdscribe" />
    </a>
    <a href='https://mofdscribe.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/mofdscribe/badge/?version=latest' alt='Documentation Status' />
    </
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://matsci.org/c/mofdscribe/56">
    <img src="https://img.shields.io/badge/matsci-discuss%20%26%20get%20help-yellowgreen" alt="Matsci">
    <a href='http://commitizen.github.io/cz-cli/'>
        <img src='https://img.shields.io/badge/commitizen-friendly-brightgreen.svg' alt='Commitizen friendly' />
    </a>
</p>

Featurizing metal-organic frameworks (MOFs) made simple! This package builds on the power of [matminer](https://hackingmaterials.lbl.gov/matminer/) to make featurization of MOFs as easy as possible. Now, you can use features that are mostly used for porous materials in the same way as all other matminer featurizers.
mofdscribe additionally includes routines that help with model validation.

## 💪 Getting Started

```python

from mofdscribe.featurizers.chemistry import RACS
from pymatgen.core import Structure

structure = Structure.from_file(<my_cif.cif>)
featurizer = RACS()
racs_features = featurizer.featurize(structure)
```

## 🚀 Installation

While we are in the process of trying to make mofdscribe work on all operating system (we're waiting for conda recipies getting merged),
it is currently not easy on Windows (and there might be potential issues on ARM-based Macs).
For this reason, we recommend installing mofdscribe on a UNIX machine.

<!-- The most recent release can be installed from
[PyPI](https://pypi.org/project/mofdscribe/) with:

```bash
$ pip install mofdscribe
``` -->
<!--

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/kjappelbaum/mofdscribe.git
``` -->

To install in development mode, use the following:

```bash
git clone git+https://github.com/kjappelbaum/mofdscribe.git
cd mofdscribe
pip install -e .
```

if you want to use all utilities, you can use the `all` extra: `pip install -e ".[all]"`

We depend on many other external tools. Currently, you need to manually install these dependencies (due to pending merges for conda-recipies):

```bash
# RASPA and Zeo++ (if you want to use energy grid/Henry coefficient and pore descriptors)
conda install -c conda-forge raspa2 zeopp-lsmo

# cgal dependency for moltda (if you want to use persistent-homology based features)
# on some systems, you might need to replace this with sudo apt-get install libcgal-dev or brew install cgal 
conda install -c conda-forge cgal dionysus

# openbabel dependency for moffragmentor (if you want to use SBU-centered features)
conda install -c conda-forge openbabel
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/kjappelbaum/mofdscribe/blob/master/CONTRIBUTING.rst) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.


### 📖 Citation

See the [ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/630d1f6f90802d52c16eceb2).

```
@article{Jablonka_2022,
	doi = {10.26434/chemrxiv-2022-4g7rx},
	url = {https://doi.org/10.26434%2Fchemrxiv-2022-4g7rx},
	year = 2022,
	month = {sep},
	publisher = {American Chemical Society ({ACS})},
	author = {Kevin Maik Jablonka and Andrew S. Rosen and Aditi S. Krishnapriyan and Berend Smit},
	title = {An ecosystem for digital reticular chemistry}
}
```


<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

### 💰 Funding

The research was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([grant agreement 666983, MaGic](https://cordis.europa.eu/project/id/666983)), by the [NCCR-MARVEL](https://www.nccr-marvel.ch/), funded by the Swiss National Science Foundation, and by the Swiss National Science Foundation (SNSF) under Grant 200021_172759.

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/kjappelbaum/mofdscribe/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
   `src/mofdscribe/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.

</details>
