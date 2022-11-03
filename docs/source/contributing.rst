Extending and contributing to mofdscribe
==========================================


Implementing a new featurizer
-----------------------------

To implement a new featurizer, you typically need to create a new class that inherits from the :py:class:`mofdscribe.featurizers.base.MOFBaseFeaturizer`. In this class, you need to implement three methods: 
:py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.featurize`,  :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.feature_labels` and :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.citation`.

The main featurization logic happens in :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.featurize`.
Your method should accept as input a :py:class:`~pymatgen.core.Structure` object and return a :py:class:`numpy.array`.
The :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.feature_labels` method should return a list of strings that describe the features returned by :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.featurize`. The number of feature names should match the number of features returned by :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.featurize` (i.e. the number of columns in the feature matrix). The :py:meth:`~mofdscribe.featurizers.base.MOFBaseFeaturizer.citation` method should return a list of strings of BibTeX citations for the featurizer.

Generally, you also want to decorate your structure with the 
decorators :py:meth:`~mofdscribe.featurizer.utils.extend.operates_on_imolecule`, :py:meth:`~mofdscribe.featurizer.utils.extend.operates_on_molecule`,  :py:meth:`~mofdscribe.featurizer.utils.extend.operates_on_structure`,  :py:meth:`~mofdscribe.featurizer.utils.extend.operates_on_istructure`. This is relevant for featurizer that operate on the building blocks and must pass the input in the right form.


Implementing a new dataset
-----------------------------

Often, you may want to use the utilities of a :py:class:`mofdscribe.datasets.dataset.AbstractStructureDataset` and the integration with the splitters, but with your custom structures and labels and not the ones shipped with mofdscribe. 

.. note:: Contribute your dataset

    Once you wrapped your dataset in a :py:class:`mofdscribe.datasets.dataset.AbstractStructureDataset`, you can contribute it to mofdscribe by opening a pull request on the mofdscribe repository. We will be happy to include it in the next release.

    This will make it easier for other researchers to build on top of your work and to compare their results with yours.
    We can then also use it to create benchmark tasks.