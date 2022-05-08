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

.. code-block:: shell

    conda install -c conda-forge cgal zeopp-lsmo raspa2

The most recent code and data can be installed directly from GitHub with:

.. code-block:: shell

    $ pip install git+https://github.com/kjappelbaum/mofdscribe.git

To install in development mode, use the following:

.. code-block:: shell

    $ git clone git+https://github.com/kjappelbaum/mofdscribe.git
    $ cd mofdscribe
    $ pip install -e .


Featurizing a MOF
------------------

.. code-block:: python

    from mofdscribe.chemistry.racs import RACS
    from pymatgen.core import IStructure

    s = IStructure.from_file(<my_cif>)
    featurizer = RACS()
    features = featurizer.featurize(s)

It is also easy to combine multiple featurizers into a single pipeline:

.. code-block:: python

    from mofdscribe.chemistry.racs import RACS
    from mofdscribe.pore.geometric_properties import PoreDiameters
    from pymatgen.core import IStructure
    from matminer.featurizers.base import MultipleFeaturizer

    s = IStructure.from_file(<my_cif>)
    featurizer = MultipleFeaturizer([RACS(), PoreDiameters()])
    features = featurizer.featurize(s)

You can, of course, also pass multiple structures to the featurizer (and the featurization is automatically parallelized via matminer):

.. code-block:: python

  s = IStructure.from_file(<my_cif>)
  s2 = IStructure.from_file(<my_cif2>)
  features = featurizer.featurize_many([s, s2])


And, clearly, you can also use the `mofdscribe` featurizers alongside ones from `matminer`:

.. code-block:: python

    from matminer.featurizers.structure import LocalStructuralOrderParams
    from mofdscribe.chemistry.racs import RACS

    featurizer = MultipleFeaturizer([RACS(), LocalStructuralOrderParams()])
    features = featurizer.featurize_many([s, s2])
