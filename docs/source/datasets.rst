Datasets in mofdscribe
=======================

StructureDatasets
------------------

The main class of datasets that mofdscribe currently provides are :py:class:`~mofdscribe.datasets.dataset.StructureDataset`s.
They bascially act as a wrapper around a collection of :py:class:`pymatgen.core.Structure` objects, but also provide some additional data such as pre-computed features and some labels.

Why use StructureDatasets?
...........................

The main reason for using :py:class:`~mofdscribe.datasets.dataset.StructureDataset`s is to provide a unified interface to different datasets (making it easy to reuse code for different datasets). That unified interface allows to use the splitters implemented in mofdscribe.

However, :py:class:`~mofdscribe.datasets.dataset.StructureDataset`s also provide some other conveniences 

- hashes for de-duplication are automatically computed if not available 
- additional metadata (e.g. publication years) is provided (if available)
- you do not need to worry about maintaining folders of different versions yourself --- mofdscribe will handle the version management for you, and you can be sure that other users of mofdscribe will use the same dataset
- makes it pretty easy to visualize a structure for a given entry

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
