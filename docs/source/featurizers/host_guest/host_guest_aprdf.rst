Guest-centered atomic-property labeled radial distribution function (APRDF)
............................................................................


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
