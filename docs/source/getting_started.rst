Getting started
==================


Installation
--------------
Do to the external dependencies, we recommend installation via conda

.. code-block:: shell 

    $ conda install -c conda-forge mofdscribe

The most recent release can be installed from
`PyPI <https://pypi.org/project/mofdscribe>`_ with:

.. code-block:: shell

    $ pip install mofdscribe

However, in this case, the following dependencies need to be manually installed (e.g. via conda):

- RASPA2
- zeo++
- cgal 

The most recent code and data can be installed directly from GitHub with:

.. code-block:: shell

    $ pip install git+https://github.com/kjappelbaum/mof-dscribe.git

To install in development mode, use the following:

.. code-block:: shell

    $ git clone git+https://github.com/kjappelbaum/mof-dscribe.git
    $ cd mof-dscribe
    $ pip install -e .


Featurizing a MOF
------------------

.. code-block:: python 

    from mofdscribe.chemistry.racs import RACS
    from pymatgen.core import IStructure 

    s = IStructure.from_file(<my_cif>)
    featurizer = RACS()
    features = featurizer.featurize(s)