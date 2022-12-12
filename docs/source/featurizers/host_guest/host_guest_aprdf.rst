Guest-centered atomic-property labeled radial distribution function (APRDF)
............................................................................

This featurizer builds on the :ref:`APRDF` featurizer, but instead of using the 
correlations between all atoms, it only considers the ones between the guest and all host atoms 
(within some cutoff distance). 

.. math::

  \operatorname{RDF}^{P}(R)=f \sum_{i, j}^{\text {all atom pairs }} P_{i} P_{j} \mathrm{e}^{-B\left(r_{i j}-R\right)^{2}}


.. featurizer::  GuestCenteredAPRDF
    :id: GuestCenteredAPRDF
    :considers_geometry: True
    :considers_structure_graph: False
    :encodes_chemistry: optionally
    :scope: global
    :scalar: False

    Based on APRDF described in [Fernandez2013]_.
