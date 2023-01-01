Shortest-Path Based Description of Building Blocks
======================================================

For certain targets, the proximity of connecting groups in a building unit 
(e.g. carboxy groups) can be interesting features. 

One way to describe this generally is to compute the distribution 
of shortest paths between special sites in the building units that
our ``moffragmentor`` package calls "binding sites" and "branching sites". 

.. featurizer::  BranchingNumHopFeaturizer
    :id: BranchingNumHopFeaturizer
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: False

.. featurizer::  BindingNumHopFeaturizer
    :id: BindingNumHopFeaturizer
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: False
