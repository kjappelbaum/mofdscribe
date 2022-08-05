Datasets in mofdscribe
=======================

StructureDatasets
------------------

The main class of datasets that mofdscribe currently provides is :py:class:`~mofdscribe.datasets.dataset.StructureDataset`.
They basically act as a wrapper around a collection of :py:class:`pymatgen.core.Structure` objects, but also provide some additional data such as pre-computed features and some labels.

Why use StructureDatasets?
...........................

The main reason for using :py:class:`~mofdscribe.datasets.dataset.StructureDataset` is to provide a unified interface to different datasets (making it easy to reuse code for different datasets). That unified interface allows to use the splitters implemented in mofdscribe.

However, :py:class:`~mofdscribe.datasets.dataset.StructureDataset` also provides some other conveniences 

- hashes for de-duplication are automatically computed if not available 
- additional metadata (e.g. publication years) is provided (if available)
- you do not need to worry about maintaining folders of different versions yourself --- mofdscribe will handle the version management for you, and you can be sure that other users of mofdscribe will use the same dataset
- makes it pretty easy to visualize a structure for a given entry
- you only need to download the data once 

.. admonition:: Where is the data? 
    :class: hint 

    The data will be downloaded into a :code:`~/.data/mofdscribe` folder. If you run into issues, you can consider deleting the folder corresponding to a specific dataset to trigger a re-download.

.. admonition::  Visualizing structures 
    :class: hint

    If you're in a notebook you can simply call :py:func:`~mofdscribe.datasets.dataset.StructureDataset.show_structure` to look at a structure.
    This can be handy if you want to explore extremes of a dataset (however, also here keep in mind that looking at the full dataset before splitting is considered data leakage).


    .. image:: figures/show_structure.png
        :width: 600
        :align: center
        :alt: show_structure
        :target: _blank


.. admonition:: Constructing a subset 
    :class: hint
    
    For some applications (e.g., nested cross-validation) you want to construct a subset of the dataset. You can do so easily using the :py:func:`~mofdscribe.datasets.dataset.StructureDataset.get_subset` function.


.. admonition:: Dataframe conventions
    :class: note 

    When we also provide a :py:class:`pandas.DataFrame` for the dataset, we follow these conventions:

    * dataframe is accessible via the :py:attr:`~mofdscribe.datasets.dataset.StructureDataset._df` attribute
    * outputs of simulations are prefixed with :code:`output`
    * features are prefixed with :code:`features`
    * additional infos such as hashes are prefixed with :code:`info`
    * if there are multiple flavors of dataset, we provide boolean masks under columns prefixed with :code:`flavor`