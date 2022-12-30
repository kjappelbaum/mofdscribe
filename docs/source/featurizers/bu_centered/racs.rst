
Revised autocorrelation functions (RACs)
.............................................

See also :ref:`RACs <RACs>`.

This featurizer is a flavor of the :ref:`RACs <RACs>` featurizer, that can split the computation over user-defined atom groups and automatically determined communities. For determining communities, we use ``networkx``'s implementation of greedy modularity maximization [NewmanModularity]_.
This community detection often corresponds to chemically meaningful parts (often ring systems) of the structure (but does not require an explicit fragmentation algorithm).

In mofdscribe, you can customize the encodings :math:`P` (using all properties that are available in our `element-coder <https://github.com/kjappelbaum/element-coder>`_ package) as well as the aggregation functions.

.. featurizer::  ModularityCommunityCenteredRACS
    :id: ModularityCommunityCenteredRACS
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: optionally
    :scope: local
    :scalar: False
    :style: only-light

    Initially described in [Janet2017]_ for metal complexes, extended to MOFs in [Moosavi2021]_.
