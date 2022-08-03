
Revised autocorrelation functions (RACs)
.............................................

Revised autocorrelation functions have originally been proposed for
metal-complexes [Janet2017]_. Autocorrelation functions have been widely used as
compact, fixed-length descriptors and are defined as

.. math::

    P_{d}=\sum_{i} \sum_{j} P_{i} P_{j} \delta\left(d_{i j}, d\right)

where :math:`P_d` is the autocorrelation for property :math:`P` at depth
:math:`d`, δ is the Dirac delta function, and :math:`d_{ij}` is the bond wise
path distance between atoms :math:`i` and :math:`j`. Janet and Kulik proposed to
constrain both the starting indices and the scopes of the
autorcorrelation functions to account for the (potentially) greater importance
of certain atoms such as the metal and its coordination sphere.

.. math::

  \underset{\text{ax / eq / all}}{\text{lc / mc}}  P_{d}^{\prime}=\sum_{i}^{l c \text {or mc scope }} \sum_{j}\left(P_{i}-P_{j}\right) \delta\left(d_{i j}, d\right)

[Moosavi2021]_ adapted this concept for MOFs and proposed to compute metal-,
ligand-, and functional-groups centered RACs.


In mofdscribe, you can customize the encodings :math:`P` (using all properties that are available in our `element-coder <https://github.com/kjappelbaum/element-coder>`_ package) as well as the aggregation functions.

.. featurizer::  RACS
    :id: RACS
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: optionally
    :scope: local
    :scalar: False
    :style: only-light

    Initially described in [Janet2017]_ for metal complexes, extended to MOFs in [Moosavi2021]_.
