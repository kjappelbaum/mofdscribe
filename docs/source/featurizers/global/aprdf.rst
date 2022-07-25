Atomic-property labeled radial distribution function (APRDF)
..............................................................

Radial distribution function (RDF) are widely used in crystallography and
molecular simulations. They describe the probability of finding a pair of atoms
separated by a certain distance. For crystalline solids, the RDF is an infinite
sequence of sharp peaks. However, the RDF contains no information about the
nature of the atoms. To introduce "chemistry" in this descriptor, Fernandez et
al. [Fernandez2013]_ proposed to use the APRDF to describe the local environment
of a given atom. The APRDF is defined as RDF weighted by the product of atomic
properties.

.. math::

  \operatorname{RDF}^{P}(R)=f \sum_{i, j}^{\text {all atom pairs }} P_{i} P_{j} \mathrm{e}^{-B\left(r_{i j}-R\right)^{2}}


.. featurizer::  APRDF
    :id: APRDF
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: optionally
    :scope: global
    :scalar: False

    Initially described in [Fernandez2013]_.
