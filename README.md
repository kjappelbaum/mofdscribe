
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
$ git clone git+https://github.com/kjappelbaum/mofdscribe.git
$ cd mofdscribe
$ pip install -e .
```

if you want to use all utilities, you can use the `all` extra: `pip install -e ".[all]"`

We depend on many other external tools. Currently, you need to manually install (due to pending merges for conda-recipies):

- `conda install -c conda-forge raspa2 zeopp-lsmo`
- `moltda` from my refactor branch https://github.com/kjappelbaum/molecule-tda/tree/refactor. Note that `moltda` depends on [`cgal`](https://anaconda.org/conda-forge/cgal)
- `moffragmentor` from my private repository  https://github.com/kjappelbaum/moffragmentor (which additionally requires `conda install -c conda-forge openbabel`)

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/kjappelbaum/mofdscribe/blob/master/CONTRIBUTING.rst) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

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
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/kjappelbaum/mofdscribe/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
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
