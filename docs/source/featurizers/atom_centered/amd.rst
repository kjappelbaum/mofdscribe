Average minimum distance
==========================

An intuitive way to describe geometries is via distances.
However, a simple list of distances is not a good descriptor as it is not permutation invariant.

[PDD]_ solves this using lexicographic sorting and [AMD]_ derives a fixed-length descriptor per structure using the average as pooling function.

In mofdscribe, we allow the user to specify a pooling function and to compute the descriptor for custom sets of elements.


.. featurizer::  AMD
    :id: AMD
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: optionally
    :scope: local
    :scalar: False

    Described by Widdowson et al. [AMD]_ [PDD]_.
