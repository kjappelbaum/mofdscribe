RDKit-derived features
========================

Via the :py:obj:`~mofdscribe.bu.rdkitadaptor.RDKitAdaptor` class you can wrap any featurizer that works on RDKit molecules into a featurizer that operates on BUS.
As an example, see how some of the featurizers below are implemented.

.. code-block:: python

    from rdkit.Chem.Descriptors3D import InertialShapeFactor as InertialShapeFactor_RDKIT

    class InertialShapeFactor(RDKitAdaptor):
        """Featurizer for the RDKit InertialShapeFactor descriptor."""

        def __init__(self):
            """Construct a new InertialShapeFactor featurizer."""
            super().__init__(InertialShapeFactor_RDKIT, ["inertial_shape_factor"])

        def citations(self) -> List[str]:
            return self.super().citations() + [
                "@incollection{Todeschini2008,"
                "doi = {10.1002/9783527618279.ch37},"
                "url = {https://doi.org/10.1002/9783527618279.ch37},"
                "year = {2008},"
                "month = may,"
                "publisher = {Wiley-{VCH} Verlag {GmbH}},"
                "pages = {1004--1033},"
                "author = {Roberto Todeschini and Viviana Consonni},"
                "title = {Descriptors from Molecular Geometry},"
                "booktitle = {Handbook of Chemoinformatics}"
                "}"
            ]

Instead of subclassing, you can also simply use the following syntax

.. code-block:: python

    from mofdscribe.bu.rdkitadaptor import RDKitAdaptor
    from rdkit.Chem.Descriptors3D import InertialShapeFactor

    my_featurizer = RDKitAdaptor(InertialShapeFactor, ["inertial_shape_factor"])


Ligand shape
.............

.. image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs10822-013-9659-1/MediaObjects/10822_2013_9659_Fig2_HTML.gif
    :alt: Ligand shape descriptors described by Wirth et al.

.. featurizer::  SphereLikeness
    :id: SphereLikeness
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

    This descriptor is computed as NPR1+NPR2-1, and has been proposed by [Wirth]_.

.. featurizer::  DiskLikeness
    :id: DiskLikeness
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

    This descriptor is computed as 2 - 2 * NPR2, and has been proposed by [Wirth]_.

.. featurizer::  RodLikeness
    :id: RodLikeness
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

    This descriptor is computed as NPR2 - NPR1, and has been proposed by [Wirth]_.


Direct RDKit ports
.....................

The following featurizers are the wrapped RDKit implementations (under the same name).

.. featurizer::  SpherocityIndex
    :id: SpherocityIndex
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  RadiusOfGyration
    :id: RadiusOfGyration
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  Asphericity
    :id: Asphericity
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  Eccentricity
    :id: Eccentricity
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  InertialShapeFactor
    :id: InertialShapeFactor
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  NPR1
    :id: NPR1
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  NPR2
    :id: NPR2
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  PMI1
    :id: PMI1
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  PMI2
    :id: PMI2
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer::  PMI3
    :id: PMI3
    :considers_geometry: True
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer:: SmartsMatchCounter
    :id: SmartsMatchCounter
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer:: AcidGroupCounter
    :id: AcidGroupCounter
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True

.. featurizer:: BaseGroupCounter
    :id: BaseGroupCounter
    :considers_geometry: False
    :considers_structure_graph: True
    :encodes_chemistry: False
    :scope: bu
    :scalar: True
