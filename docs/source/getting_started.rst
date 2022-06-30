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

However, in this case, the following dependencies need to be manually installed
(e.g. via conda):

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

You can, of course, also pass multiple structures to the featurizer (and the
featurization is automatically parallelized via matminer):

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


If you use the :code:`zeo++` or :code:`raspa2` packages, you can customize the temporary
directory used by the featurizers by exporting :code:`MOFDSCRIBE_TEMPDIR`. If you do
not specify the temporary directory, the default is the current working
directory.

Using a reference dataset
--------------------------

mofdscribe contains some de-duplicated structure datasets (with labels) that can
be useful to make machine learning studies more comparable. To use a reference
dataset, you simply need to instantiate the corresponding object.

.. code-block:: python

        from mofdscribe.datasets import CoRE, QMOF
        qmof = QMOF() # will use no labels and the latest version of the dataset

Upon first use this will download the datasets into a folder
`~/.data/mofdscribe` in your home directory. In case of corruption or problems
you hence can also try removing the subfolders. The package should automatically
download the missing files. Note that the currently implemented datasets are
loaded completely into memory. On modern machines this should not be a problem,
but it might be if you are resource constrained.

You get also get a specific entry with

.. code-block:: python

    qmof.get_structure(1)


Using splitters
-----------------

For model validation it is important to use stringent splits into folds. In many
cases, a random split is not ideal for materials discovery application, where
extrapolation is often more relevant than interpolation. To model extrapolative
behavior, one can some of the splitting strategies implemented in mofdscribe.
They all assume :py:meth:`~mofdscribe.datasets.dataset.StructureDataset` as
input.

.. code-block:: python

    from mofdscribe.splitters import TimeSplitter, HashSplitter
    from mofdscribe.datasets import CoRE

    ds = CoRE()

    splitter = TimeSplitter()

    train_idx, valid_idx, test_idx = splitter.train_valid_test_split(ds,
        train_frac=0.7, valid_frac=0.1)


Using metrics 
-----------------

For making machine learning comparable, it is important to report reliable metrics. 
mofdscribe implements some helpers to make this easier.

One interesting metric is the adversarial validation score, which can be a surrogate for how different two datasets, e.g. a train and a test set, are. Under the hood, this is implemented as a classifier that attempts to learn to distinguish the two datasets. If the two datasets are indistinguishable, the classifier will have a ROC-AUC of 0.5.

.. code-block:: python

    from mofdscribe.metrics import AdverserialValidator
    from mofdscribe.datasets import CoRE
    from mofdscribe.splitters import RandomSplitter

    FEATURES = ["Di", "Df", "Dif", "density [g/cm^3]",]

    ds = CoRE()    
    train_idx, test_idx = RandomSplitter().train_test_split(ds)

    adversarial_validation_scorer = AdverserialValidator(ds._df.iloc[train_idx][FEATURES], 
        ds._df.iloc[test_idx][FEATURES])

    adversarial_validation_scorer.score().mean()

However, you cannot only measure how different two datasets are, but also quantify how well your model does. A handy helper function 
is :py:meth:`~mofdscribe.metrics.regression.get_regression_metrics`.

.. code-block:: python
    
    from mofdscribe.metrics import get_regression_metrics
    
    metrics = get_regression_metrics(predictions, labels)

Which returns an object with the most relevant regression metrics.

Running a benchmark
----------------------


Referencing datasets and featurizers
--------------------------------------

If you use a dataset or featurizers please cite all the references you find in
the `citations` property of the featurizer/dataset.
