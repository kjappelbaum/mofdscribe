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


Using a reference dataset
--------------------------

mofdscribe contains some de-duplicated structure datasets (with labels) that can be useful to make machine learning studies more comparable.
To use a reference dataset, you simply need to instantiate the corresponding object.

.. code-block:: python

        from mofdscribe.datasets import QMOFElectronic, CoREGas
        qmof_electronic = QMOFElectronic() # will use no labels and the latest version of the dataset

Upon first use this will download the datasets into a folder `~/.data/mofdscribe` in your home directory.
In case of corruption or problems you hence can also try removing the subfolders. The package should automatically download the missing files.
Note that the currently implemented datasets are loaded completely into memory. On modern machines this should not be a problem, but it might be if you are resource constrained.

:class:`MOFStructureDataSet` can be iterated over to get the structures and their labels:

.. code-block:: python

        for structure, label in qmof_electronic:
            print(structure, label)

but you get also get a specific entry with 

.. code-block:: python 

    qmof_electronic.get_structure(1)

Referencing datasets and featurizers
--------------------------------------

If you use a dataset or featurizers please cite all the references you find in the `citations` property of the featurizer/dataset.
