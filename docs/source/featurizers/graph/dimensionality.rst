Dimensionality 
..................

Returns the dimensionality of the structure. This measure is based on [LarsenDimensionality]_, where the structure graph is analyzed. 

In the case of MOFs, rod like structures are considered 1D, sheet like structures are considered 2D, and 3D structures are considered 3D.

This can be interesting for the metal nodes, where the typical SBUs such as Cu-paddlewheels are 0D. However, many well-known MOFs such as MOF-74 have infinite rod nodes, that this featurizer would consider 1D.

.. featurizer::  Dimensionality
    :id: Dimensionality
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: false
    :scope: global
    :scalar: True

    Returns the dimensionality of the structure. This measure is based on [LarsenDimensionality]_, where the structure graph is analyzed.