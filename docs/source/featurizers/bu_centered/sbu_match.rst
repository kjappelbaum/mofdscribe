BU Match
==============

This featurizer is inspired by mismatch measures in MOF building tools such as `pormake <https://github.com/Sangwon91/PORMAKE>`_.
The idea is to measure the `Kabsch RMSD <https://en.wikipedia.org/wiki/Kabsch_algorithm>`_ between the optimal node embedding and the actual coordinates of a building block.


.. featurizer::  BUMatch
    :id: BUMatch
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True
