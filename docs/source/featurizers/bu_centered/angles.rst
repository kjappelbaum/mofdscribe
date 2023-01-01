Angle-based description of BU shape 
=======================================

The following featurizers compute the angles between all pairs of atoms in a building block. 
We always compute the angles A-COM-B, where COM is the center of mass of the building block.

Given the distribution of the angles, we can compute fixed-length descriptors by either converting
the distribution to a histogram or computing some statistics (mean, standard deviation, etc.) of the distribution.

.. featurizer::  PairWiseAngleHist
    :id: PairWiseAngleHist
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: False
    :scope: bu
    :scalar: False

.. featurizer::  PairWiseAngleStats
    :id: PairWiseAngleStats
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: False
    :scope: bu
    :scalar: False
