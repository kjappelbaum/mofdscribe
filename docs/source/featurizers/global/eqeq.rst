
Partial charge statistics and histogram
.............................................

The nature of the atoms and their coordination environment dictate the partial
charge distribution in a structure. Hence, they can be used as a descriptor for
the "chemistry" of a structure. Since the number of atoms in a structure is
not fixed, the partial charges cannot directly be used as a (fixed-length)
descriptor. [Moosavi2021]_ used the minimum and maximum partial charges in a
structure, but one can also compute other statistics such as the mean, range,
and standard deviation or create a histogram.

.. featurizer::  PartialChargeHistogram
    :id: PartialChargeHistogram
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: True
    :scope: global
    :scalar: False

    We use the EqEq implementation described in [Wilmer2012]_, [Ongari2017]_ to compute the charges.


.. featurizer::  PartialChargeStats
    :id: PartialChargeStats
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: True
    :scope: global
    :scalar: False

    We use the EqEq implementation described in [Wilmer2012]_, [Ongari2017]_ to compute the charges.
