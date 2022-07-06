Featurizers
===================
.. Potential additional categories: High-dimensional? Models pores?

Many of the descriptors implemented in mofdscribe have been discussed in our
`2020 Chem. Rev. article
<https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00004>`_.


.. needtable::
   :types: featurizer
   :style: datatable
   :class: table-striped
   :columns: id, considers_geometry, considers_structure_graph, encodes_chemistry, scalar, scope

.. list-table:: Overview of implemented global featurizers
   :widths: 25 20 20 20 20 25
   :header-rows: 1

   * - Name
     - Contains chemistry information
     - Derived using geometry information
     - Derived using structure graph
     - representative reference
   * - `APRDF <:py:class:`mofdscribe.featurizers.chemistry.aprdf.APRDF>`_
     - ✅
     - ✅
     - ❌
     - [Fernandez2013]_
   * - `EnergyGridHistogram <:py:class:`mofdscribe.featurizers.chemistry.energygrid.EnergyGridHistogram>`_
     - ✅
     - ✅
     - ❌
     - [Bucior2019]_
   * - `Henry <:py:class:`mofdscribe.featurizers.chemistry.henry.Henry>`_
     - ✅
     - ✅
     - ❌
     -
   * - `PartialChargeStats <:py:class:`mofdscribe.featurizers.chemistry.partialchargestats.PartialChargeStats>`_
     - ✅
     - ✅
     - ❌
     - [Moosavi2021]_ [Ongari2019]_ [Wilmer2012]_
   * - `PartialChargeHistogram <:py:class:`mofdscribe.featurizers.chemistry.partialchargehistogram.PartialChargeHistogram>`_
     - ✅
     - ✅
     - ❌
     - [Ongari2019]_ [Wilmer2012]_
   * - `AMD <:py:class:`mofdscribe.featurizers.chemistry.amd.AMD>`_
     - (✅, optionally)
     - ✅
     - ❌
     - [Widdowson2022]_
   * - `PoreDiameters <:py:class:`mofdscribe.featurizers.pore.geometric_properties.PoreDiameters>`_
     - ❌
     - ✅
     - ❌
     - [Willems2011]_
   * - `SurfaceArea <:py:class:`mofdscribe.featurizers.pore.geometric_properties.SurfaceArea>`_
     - ❌
     - ✅
     - ❌
     - [Willems2011]_
   * - `AccessibleVolume <:py:class:`mofdscribe.featurizers.pore.geometric_properties.AccessibleVolume>`_
     - ❌
     - ✅
     - ❌
     - [Willems2011]_ [Ongari2017]_
   * - `RayTracingHistogram <:py:class:`mofdscribe.featurizers.pore.geometric_properties.RayTracingHistogram>`_
     - ❌
     - ✅
     - ❌
     - [Willems2011]_ [Pinheiro2013]_
   * - `PoreSizeDistribution <:py:class:`mofdscribe.featurizers.pore.geometric_properties.PoreSizeDistribution>`_
     - ❌
     - ✅
     - ❌
     - [Willems2011]_ [Pinheiro2013]_
   * - `VoxelGrid <:py:class:`mofdscribe.featurizers.pore.voxelgrid.VoxelGrid>`_
     - ✅
     - ✅
     - ❌
     -
   * - `PHImage <:py:class:`mofdscribe.featurizers.topology.ph_image.PHImage>`_
     - ✅ (optionally)
     - ✅
     - ❌
     - [Adams2017]_ [Krishnapriyan2021]_ [Krishnapriyan2020]_
   * - `PHVect <:py:class:`mofdscribe.featurizers.topology.ph_vect.PHVect>`_
     - ✅ (optionally)
     - ✅
     - ❌
     - [Perea]_ [Tymochko]_
   * - `PHStats <:py:class:`mofdscribe.featurizers.topology.ph_stats.PHStats>`_
     - ✅ (optionally)
     - ✅
     - ❌
     -
   * - `PHHist <:py:class:`mofdscribe.featurizers.topology.ph_hist.PHHist>`_
     - ✅ (optionally)
     - ✅
     - ❌
     -



.. list-table:: Overview of implemented atom-centered featurizers
   :widths: 25 20 20 20 20 25
   :header-rows: 1

   * - Name
     - Contains chemistry information
     - Derived using geometry information
     - Derived using structure graph
     - representative reference
   * - `RACS <:py:class:`mofdscribe.chemistry.racs.RACS>`_
     - ✅
     - ❌
     - ✅
     - [Moosavi2021]_
   * - `AtomCenteredPH <:py:class:`mofdscribe.featurizers.topology.atom_centered_ph.AtomCenteredPH>`_
     - ✅ (optionally)
     - ✅
     - ❌
     - [Jiang2021]_



.. list-table:: Overview of implemented SBU-centered featurizers
   :widths: 25 20 20 20 20 25
   :header-rows: 1

   * - Name
     - Contains chemistry information
     - Derived using geometry information
     - Derived using structure graph
     - representative reference
   * - `PairwiseDistanceHist <:py:class:`mofdscribe.featurizers.sbu.distance_hist_featurizer.PairwiseDistanceHist>`_
     - ❌
     - ✅
     - ❌
     - 
   * - `PairwiseDistanceStats <:py:class:`mofdscribe.featurizers.sbu.distance_stats_featurizer.PairwiseDistanceStats>`_
     - ❌
     - ✅
     - ❌
     - 
   * - `LSOP <:py:class:`mofdscribe.featurizers.sbu.lsop_featurizer.LSOP>`_
     - ❌
     - ✅
     - ❌
     - 
   * - `NConf20 <:py:class:`mofdscribe.featurizers.sbu.nconf20_featurizer.NConf20>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `Asphericity <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.Asphericity>`_
     - ❌
     - ✅
     - ✅
     -   
   * - `Eccentricity <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.Eccentricity>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `InertialShapeFactor <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.InertialShapeFactor>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `NPR1 <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.NPR1>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `NPR2 <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.NPR2>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `PMI1 <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.PMI1>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `PMI2 <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.PMI2>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `PMI3 <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.PMI3>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `RadiusOfGyration <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.RadiusOfGyration>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `SpherocityIndex <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.SpherocityIndex>`_
     - ❌
     - ✅
     - ✅
    - 
   * - `RodLikeness <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.RodLikeness>`_
     - ❌
     - ✅
     - ✅
    - 
   * - `DiskLikeness <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.DiskLikeness>`_
     - ❌
     - ✅
     - ✅
     - 
   * - `SphereLikeness <:py:class:`mofdscribe.featurizers.sbu.shape_featurizer.SphereLikeness>`_
     - ❌
     - ✅
     - ✅
    - 



Atom-centered featurizers
------------------------------

A key approximation for machine learning in chemistry is the locality
approximation. Effectively, this allows to train models on small fragments which
then (hopefully) can be used to predict the properties of larger structures.


.. toctree::
    :glob:
    :maxdepth: 1  

    featurizers/atom_centered/*



Global featurizers
--------------------
In particular for porous materials, some properties are not local. For
instance, the pore geometry (key for gas adsorption) cannot be captured by
descriptor that only considers the local environment (of e.g., 3 atoms). For this reason it can make sense to also consider features that consider the full structure as a whole.

.. toctree::
    :glob:
    :maxdepth: 1  

    featurizers/global/*



SBU-centered featurizers
-----------------------------

Reticular chemistry describes materials built via a tinker-toy approach.
Hence, a natural approach is to focus on the building blocks. 

mofdscribe can compute descriptors that are SBU-centred, for instance, using RDKit descriptors on the building blocks. 

For this, you can either provide your building blocks that you extracted with any of the available tools, or use our integration with our `moffragmentor <https://github.com/kjappelbaum/moffragmentor>`_ package. In this case, we will fragment the MOF into its building blocks and then compute the features for each building block and let you choose how you want to aggregate them.


.. toctree::
    :glob:
    :maxdepth: 1  

    featurizers/sbu_centered/*
