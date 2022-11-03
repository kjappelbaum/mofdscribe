Getting started
==================


Featurizing a MOF
------------------

.. code-block:: python

    from mofdscribe.chemistry.racs import RACS
    from pymatgen.core import Structure

    s = Structure.from_file(<my_cif>)
    featurizer = RACS()
    features = featurizer.featurize(s)

.. admonition:: mofdscribe base classes
    :class: hint

    Most featurizers in mofdscribe inherit from :py:class:`~mofdscribe.featurizers.base.MOFBaseFeaturizer`.
    This class can also handle the conversion to primitive cells if you pass :code:`primitive=True` to the
    constructor. This can be useful to save computational time but also make it possible to, e.g., 
    use the :code:`sum` aggregation.

    To avoid re-computation of the primitive cell, you should use the :py:class:`~mofdscribe.featurizers.base.MOFMultipleFeaturizer`
    for combining multiple featurizers. This will accept a keyword argument :code:`primitive=True` in the constructor 
    and then compute the primitive cell once and use it for all the featurizers.

It is also easy to combine multiple featurizers into a single pipeline:

.. code-block:: python

    from mofdscribe.chemistry.racs import RACS
    from mofdscribe.pore.geometric_properties import PoreDiameters
    from pymatgen.core import Structure
    from mofdscribe.featurizers.base import MOFMultipleFeaturizer

    s = Structure.from_file(<my_cif>)
    featurizer = MOFMultipleFeaturizer([RACS(), PoreDiameters()])
    features = featurizer.featurize(s)

You can, of course, also pass multiple structures to the featurizer (and the
featurization is automatically parallelized via matminer):

.. code-block:: python

  s = Structure.from_file(<my_cif>)
  s2 = Structure.from_file(<my_cif2>)
  features = featurizer.featurize_many([s, s2])


And, clearly, you can also use the `mofdscribe` featurizers alongside ones from `matminer`:

.. code-block:: python

    from matminer.featurizers.structure import LocalStructuralOrderParams
    from mofdscribe.chemistry.racs import RACS

    featurizer = MOFMultipleFeaturizer([RACS(), LocalStructuralOrderParams()])
    features = featurizer.featurize_many([s, s2])


If you use the :code:`zeo++` or :code:`raspa2` packages, you can customize the temporary
directory used by the featurizers by exporting :code:`MOFDSCRIBE_TEMPDIR`. If you do
not specify the temporary directory, the default is the current working
directory.

.. admonition:: More examples
    :class: info 

    You can find more examples of how to featurize MOFs in the `featurize.ipynb`
    and notebook in the `examples folder <https://github.com/kjappelbaum/mofdscribe/tree/main/examples>`_.


Using a reference dataset
--------------------------

mofdscribe contains some de-duplicated structure datasets (with labels) that can
be useful to make machine learning studies more comparable. To use a reference
dataset, you simply need to instantiate the corresponding object.

.. code-block:: python

        from mofdscribe.datasets import CoRE, QMOF
        qmof = QMOF() # will use no labels and the latest version of the dataset

Upon first use this will download the datasets into a folder
:code:`~/.data/mofdscribe` in your home directory. In case of corruption or problems
you hence can also try removing the subfolders. The package should automatically
download the missing files. Note that the currently implemented datasets are
loaded completely into memory. On modern machines this should not be a problem,
but it might be if you are resource constrained.

You get also get a specific entry with

.. code-block:: python

    qmof.get_structure(1)

mofdscribe tries to reduce the potential for data leakage by dropping duplicates.
However, it is not trivial to define what is a duplicate. See :ref:`dataleakage`
for more information.

Using splitters
-----------------

For model validation it is important to use stringent splits into folds. In many
cases, a random split is not ideal for materials discovery application, where
extrapolation is often more relevant than interpolation.
To model extrapolative behavior,
one can some splitting strategies implemented in mofdscribe.
They all assume :py:meth:`~mofdscribe.datasets.dataset.StructureDataset` as
input.

.. code-block:: python

    from mofdscribe.splitters import TimeSplitter, HashSplitter
    from mofdscribe.datasets import CoRE

    ds = CoRE()

    splitter = TimeSplitter(ds)

    train_idx, valid_idx, test_idx = splitter.train_valid_test_split(train_frac=0.7, valid_frac=0.1)


All splitters are implemented based on :py:meth:`~mofdscribe.splitters.splitters.BaseSplitter`.
If you want to implement a custom grouping or stratification strategy, you'll need to implement the

* :code:`_get_stratification_col`: Should return an ArrayLike object of floats, categories, or ints.
            If it is categorical data, the :code:`BaseSplitter` will handle the discretization.
* :code:`_get_groups`: Should return an ArrayLike object of categories (integers or strings)

methods.

Using metrics
-----------------

For making machine learning comparable, it is important to report reliable metrics.
mofdscribe implements some helpers to make this easier.

One interesting metric is the adversarial validation score, which can be a surrogate for how different two datasets, e.g. a train and a test set, are. Under the hood, this is implemented as a classifier that attempts to learn to distinguish the two datasets. If the two datasets are indistinguishable, the classifier will have a ROC-AUC of 0.5.

.. code-block:: python

    from mofdscribe.metrics import AdverserialValidator
    from mofdscribe.datasets import CoRE
    from mofdscribe.splitters import RandomSplitter

    ds = CoRE()

    FEATURES = list(ds.available_features)

    train_idx, test_idx = RandomSplitter(ds).train_test_split(fract_train=0.8)

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

The benchmarks will run k=5-fold cross validation on the dataset. We chose this over a single split, because this is more robust to randomness (and gives at least some indication of the variance of the estimate).

.. admonition:: OOD vs ID
    :class: info

    Most benchmarks come in OOD and ID versions.
    OOD indicates out-of-distribution, and typically involves grouping on a key feature (e.g. density).
    ID indicates in-distribution, and typically is stratified on the target variable.

.. admonition:: Why k-fold CV?
    :class: info

    For the benchmarks we decided to use k-fold cross validation.
    While this is clearly more expensive than a simple holdout split, splits need to be performed multiple
    times as ML models are unstable [Lones]_.  This is in particular the case for the relatively small
    datasets we work with in digital reticular chemistry (for larger datasets repeated holdout splits are less of a problem).
    One could add more rigor using repeated k-fold cross validation. However, this would result in a large
    computational overhead.
    Note that the choice of the k is not trivial, and k=5 is a pragmatic choice, for more details see [Raschka]_.

    Also note that the errorbars one estimates via the standard error of k-fold crossvalidation 
    are often too small. [Varoquaux]_ However, as [Varoquaux]_ writes

        Cross-validation is not a silver bullet. However, it is the best tool available, because
        it is the only non-parametric method to test for model generalization.

For running a benchmark with your model, your model must be in the form of a class with `fit(idx, structures, y)` and `predict(idx, structures)` methods, for example

.. code-block:: python

    class MyDummyModel:
        """Dummy model."""

        def __init__(self, lr_kwargs: Optional[Dict] = None):
            """Initialize the model.

            Args:
                lr_kwargs (Optional[Dict], optional): Keyword arguments
                    that are passed to the linear regressor.
                    Defaults to None.
            """
            self.model = Pipeline(
                [("scaler", StandardScaler()), ("lr", LinearRegression(**(lr_kwargs or {})))]
            )

        def featurize(self, s: Structure):
            """You might want to use a lookup in some dataframe instead.

            Or use some mofdscribe featurizers.
            """
            return s.density

        def fit(self, idx, structures, y):
            x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
            self.model.fit(x, y)

        def predict(self, idx, structures):
            x = np.array([self.featurize(s) for s in structures]).reshape(-1, 1)
            return self.model.predict(x)

.. admonition::  Use dataset in model
    :class: hint

    If you want to use the dataset in your model class, you might find the :code:`patch_in_ds` 
    keyword argument of the :py:class:`~mofdscribe.bench.mofbench.MOFBench` class useful. 
    This will make the dataset available to your model under the :code:`ds` attribute.

.. admonition:: Logging metadata 
    :class: hint

    If you want to log any additional information during the fitting process, for instance hyperparameters, you can do so using the :py:meth:`~mofdscribe.bench.mofbench.MOFBench.log` method, that we also patch into your model. 

    That is, your model will have a :code:`log` method to which you can pass a dictionary that will be appended to a list that will appear in the report.
    In this way, for instance, you can record hyperparameters or other information in each fold.


If you have a model in this form, you can use a bench class.

.. code-block:: python

    from mofdscribe.bench.logKHCO2 import LogkHCO2IDBench

    bench = LogkHCO2IDBench(MyDummyModel(), name='My great model')
    report = bench.bench()
    report.save_json(<directory>)
    report.save_rst(<directory>)

You can test this using some dummy models implemented in mofdscribe

.. code-block:: python

    from mofdscribe.bench.dummy_models import DensityRegressor

    logkHCO2_interpolation_density = LogkHCO2IDBench(
        DensityRegressor(),
        version="v0.0.1",
        name="linear density",
        features="density",
        model_type="linear regression /w polynomial features",
        implementation="mofdscribe",
        reference="mofdscribe",
    )

.. admonition:: Reference in BibTeX format
    :class: hint

    If you provide your reference in BibTeX format, it will appear in a copyable text box in the documentation. That is, it is super easy for others to cite you!

For testing purposes, you can set :code:`debug=True` in the constructors of the benchmark classes.

Which will generate a report file that you can use to make a pull request for adding your model to the leaderboard.

For this:

1. Fork the repository.
2. Make a new branch (e.g. named :code:`add_{modelname}`).
3. Add your :code:`.json` and :code:`.rst` files to the corresponding :code:`bench_results` sub folder. Do not change the name of the file, it will be used as unique identifier.
4. Push your branch to the repository.
5. Make a pull request.

Upon your PR, a pull request will ask one of the maintainers for approval for a rebuild of the leaderboard. Once we checked that you include all the important parts and some additional context (e.g. link to an implementation), your model will appear on the leaderboard.

.. admonition:: More examples
    :class: info

    You can find more examples of how to build benchmarks in the `hyperparameter_optimization_in_bench.ipynb`
    and `add_model_to_leaderboard.ipynb` notebooks in the `examples folder <https://github.com/kjappelbaum/mofdscribe/tree/main/examples>`_.

.. admonition:: Do not look at the dataset!
    :class: warning

    Do not perform hyper-parameter optimization (or model selection) on the dataset used for the benchmark
    *outside* the bench loop. This is data leakage.

    If you need to perform hyper-parameter optimization, use an approach such as nested-cross validation
    in the bench loop.
    Only this allows for fair comparison and only this allows others to reproduce the
    hyperparameter selection (and, hence, use "fair" hyperparameters when they compare their model with your model as a baseline).

Referencing datasets and featurizers
--------------------------------------

If you use a dataset or featurizers please cite all the references you find in
the `citations` property of the featurizer/dataset.


