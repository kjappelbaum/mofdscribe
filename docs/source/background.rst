Background
===================
.. Potential additional categories: High-dimensional? Models pores?

Many of the descriptors implemented in mofdscribe have been discussed in our `2020 Chem. Rev. article <https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00004>`_.

.. list-table:: Overview of implemented featurizers
   :widths: 25 20 20 20 20 25
   :header-rows: 1

   * - Name
     - Assumes locality
     - Contains chemistry information
     - Derived using geometry information
     - Derived using structure graph
     - representative reference
   * - `RACS <:py:class:`mofdscribe.chemistry.racs.RACS>`_
     - ✅
     - ✅
     - ❌
     - ✅
     - [Moosavi2021]_
   * - `APRDF <:py:class:`mofdscribe.chemistry.aprdf.APRDF>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - [Fernandez2013]_
   * - `EnergyGridHistogram <:py:class:`mofdscribe.chemistry.energygrid.EnergyGridHistogram>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - [Bucior2019]_
   * - `PartialChargeStats <:py:class:`mofdscribe.chemistry.partialchargestats.PartialChargeStats>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - [Moosavi2021]_ [Ongari2019]_ [Wilmer2012]_
   * - `PartialChargeHistogram <:py:class:`mofdscribe.chemistry.partialchargehistogram.PartialChargeHistogram>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - [Ongari2019]_ [Wilmer2012]_
   * - `PoreDiameters <:py:class:`mofdscribe.pore.geometric_properties.PoreDiameters>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - [Willems2011]_
   * - `SurfaceArea <:py:class:`mofdscribe.pore.geometric_properties.SurfaceArea>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - [Willems2011]_
   * - `AccessibleVolume <:py:class:`mofdscribe.pore.geometric_properties.AccessibleVolume>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - [Willems2011]_ [Ongari2017]_
   * - `RayTracingHistogram <:py:class:`mofdscribe.pore.geometric_properties.RayTracingHistogram>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - [Willems2011]_ [Pinheiro2013]_
   * - `PoreSizeDistribution <:py:class:`mofdscribe.pore.geometric_properties.PoreSizeDistribution>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - [Willems2011]_ [Pinheiro2013]_
   * - `PHImage <:py:class:`mofdscribe.topology.ph_image.PHImage>`_
     - ❌
     - ✅ (optionally)
     - ✅
     - ❌
     - [Adams2017]_ [Krishnapriyan2021]_ [Krishnapriyan2020]_
   * - `PHVect <:py:class:`mofdscribe.topology.ph_vect.PHVect>`_
     - ❌
     - ✅ (optionally)
     - ✅
     - ❌
     - [Perea]_ [Tymochko]_


Pore descriptors
-------------------

Scalars describing the pore geometry
.........................................
For describing the pore geometry, we heavily rely on methods implemented in the `zeopp <http://www.zeoplusplus.org/>`_ package.

The most commonly used pore-geometry descriptors (surface areas, probe accessible pore volumes, ...) are computed with a probe approach.
By means of "simple" geometry analysis one can also extract pore radii.


.. image:: http://www.zeoplusplus.org/spheres.png
  :width: 200
  :alt: Pore diameters.


Histograms
...............

An alternative to scaler descriptors are "summaries" of the pore geometry in histograms.
One approach for instance is to shoot random rays (orange in the figure below) through the structure and use the length of the ray between the points where it intersects the pores as ray length. If one performs this experiment often, one can summarize the observed ray lengths in a histogram.

.. image:: figures/rays.png
  :width: 200
  :alt: Shooting rays through pores. Figure modified from http://iglesia.cchem.berkeley.edu/Publications/2013%20Pinheiro_PSD%20v%20Ray%20histograms_J%20Mol%20Graph%20Mod%2044%20(2013)%20208.pdf

Another, quite sparse, alternative is to use the pore size distribution (PSD) of the structure. That is, PSD histograms measure what fraction of the void space volume corresponds to certain pore sizes. One might also use this as cumulative distribution or take the derivative.



"Chemistry" descriptors
--------------------------

A key approximation for machine learning in chemistry is the locality approximation. Effectively, this allows to train models on small fragments which then (hopefully) can be used to predict the properties of larger structures.
However, in particular for porous materials, some properties are not local. For instance, the pore geometry (key for gas adsorption) cannot be captured by descriptor that only considers the local environment (of e.g., 3 atoms).


Descriptors assuming locality
..................................

Revised autocorrelation functions (RACs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Revised autocorrelation functions have originally been proposed for metal-complexes [Janet2017]_. Autocorrelation functions have been widely used as compact, fixed-length descriptors and are defined as

.. math::

    P_{d}=\sum_{i} \sum_{j} P_{i} P_{j} \delta\left(d_{i j}, d\right)

where :math:`P_d` is the autocorrelation for property :math:`P` at depth :math:`d`, δ is the Dirac delta function, and :math:`d_{ij}` is the bondwise path distance between atoms :math:`i` and :math:`j`. Janet and Kulik proposed to constrain both the starting indices as well as the scopes of the autorcorrelation functions to account for the (potentially) greater importance of certain atoms such as the metal and its coordination sphere.

.. math::

  \underset{\text{ax / eq / all}}{\text{lc / mc}}  P_{d}^{\prime}=\sum_{i}^{l c \text {or mc scope }} \sum_{j}\left(P_{i}-P_{j}\right) \delta\left(d_{i j}, d\right)

[Moosavi2021]_ adapted this concept for MOFs and proposed to compute metal-, ligand-, and functional-groups centred RACs.

Non-local descriptors
..........................

Atomic-property labeled radial distribution function (APRDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Radial distribution function (RDF) are widely used in crystallography and molecular simulations. They describe the probability of finding a pair of atoms separated by a certain distance. For crystalline solids, the RDF is an infinite sequence of sharp peaks.
However, the RDF contains no information about the nature of the atoms. To introduce "chemistry" in this descriptor, Fernandez et al. [Fernandez2013]_ proposed to use the APRDF to describe the local environment of a given atom. The APRDF is defined as RDF weighted by the product of atomic properties.

.. math::

  \operatorname{RDF}^{P}(R)=f \sum_{i, j}^{\text {all atom pairs }} P_{i} P_{j} \mathrm{e}^{-B\left(r_{i j}-R\right)^{2}}


Partial charge statistics and histogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The nature of the atoms and their coordination environment dictate the partial charge distribution in a structure.
Hence, they can be used as a descriptor for the "c  hemistry" of a structure. Since the number of atoms in a structure is not fixed, the partial charges cannot directly be used as a (fixed-length) descriptor. [Moosavi2021]_ used the minimum and maximum partial charges in a structure, but one can also compute other statistics such as the mean, range, and standard deviation or create a histogram.


Energy grid histogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In molecular simulations, the interactions between atoms are described using intermolecular potentials such as the Lennard-Jones potential.
Since this potential must be frequently evaluated one can save computational cost by pre-computing the potentials on a grid.
The grids themselves are not necessarily fixed-length, and typically high-dimensional. Therefore, they are not directly used as descriptors.
Again, one can solve this problem by "summarizing" the grid in form of a histogram.


.. figure:: figures/energygrid.svg
  :width: 500
  :alt: Energy grid histogram.

  Converting MOF structures into energy grids and using them as descriptors in form of histograms. Figure taken from [Bucior2019_].


Topological descriptors
-------------------------
For many applications of porous materials the _shape_ of the material, e.g., the pore shape, is relevant for the application.
Topology is the branch of mathematics that deals with shapes and one of the most widely used topological techniques to describe shapes is known as persistent homology.

Formally speaking, persistent homology tracks the changes of homology groups in a filtration. This becomes quite clear in the following example.
In this figure, we perform a filtration and record the result in a persistence diagram. To make the filtration, we simply start increasing the "radius" of the atoms in the structure. Then we track when certain shapes (e.g., rings) appear and disappear. The "birth" and "death" of a shape is recorded in the diagram with a bar starting at the birth time and ending at the death time (e.g. in Angstrom).

.. image:: figures/ExamplePersistenceBalls3.svg
  :width: 400
  :alt: Illustration of filtration of a point cloud.


Vectorizing persistence diagrams
..................................
For many machine learning models, fixed length vectors are required.  Persistence diagrams, however, are not fixed length. In `mofdscribe` we provide two methods to vectorize persistence diagrams.

Persistence images
~~~~~~~~~~~~~~~~~~~
A method that has been used before for porous materials are persistence diagrams that have been introduced by Adams et al. in [Adams2017]_.
The idea here is to replace the points on a persistence diagram by a Gaussian (and also add a weighting function).

Gaussian mixture components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unexplored for porous materials is to use Gaussian mixture models to vectorize persistence diagrams. The idea is to train a Gaussian mixture model on a training set of persistence diagrams and then use the model to vectorize a test set of persistence diagrams (using the weighted maximum likelihood estimate of the mixture weights as vector components). [Perea]_ [Tymochko]_


References
--------------

.. [Adams2017] `Journal of Machine Learning Research 18 (2017) 1-35 <https://jmlr.csail.mit.edu/papers/volume18/16-337/16-337.pdf>`_

.. [Perea] `Jose A. Perea, Elizabeth Munch, Firas A. Khasawneh, Approximating Continuous Functions on Persistence Diagrams Using Template Functions, arXiv:1902.07190 <https://arxiv.org/abs/1902.07190>`_

.. [Tymochko] `Sarah Tymochko, Elizabeth Munch, Firas A. Khasawneh, Adaptive Partitioning for Template Functions on Persistence Diagrams, arXiv:1910.08506v1 <https://arxiv.org/abs/1910.08506v1>`_

.. [Moosavi2021] `Moosavi et al., Nature Communications 2021 <https://www.nature.com/articles/s41467-020-17755-8>`_

.. [Fernandez2013] `Fernandez et al., J. Phys. Chem. C. 2013 <https://pubs.acs.org/doi/full/10.1021/jp404287t>`_

.. [Bucior2019] `Bucior et al.,  Mol. Syst. Des. Eng. 2019 <https://pubs.rsc.org/en/content/articlelanding/2019/me/c8me00050f>`_

.. [Willems2011] `Willems et al., Microporous and Mesoporous Materials, 149 (2012) 134-141 <http://www.sciencedirect.com/science/article/pii/S1387181111003738>`_

.. [Pinheiro2013] `Pinheiro et al., Journal of Molecular Graphics and Modeling 2013, 44, 208-219 <http://www.sciencedirect.com/science/article/pii/S109332631300096X?via%3Dihub>`_

.. [Ongari2017] `Ongari et al., Langmuir 2017, 33, 14529-14538 <https://pubs.acs.org/doi/10.1021/acs.langmuir.7b016824>`_

.. [Krishnapriyan2020] `Krishnapriyan et al., J. Phys. Chem. C 2020, 124, 9360–9368 <https://www.nature.com/articles/s41598-021-88027-8>`_

.. [Krishnapriyan2021] `Krishnapriyan et al., Scientific Reports 2021, 11, 8888 <https://www.nature.com/articles/s41598-021-88027-8>`_

.. [Janet2017] `Janet, J. P.; Kulik, H. J. J. Phys. Chem. A 2017, 121 (46), 8939–8954 <https://doi.org/10.1021/acs.jpca.7b08750>`_

.. [Ongari2019] `Ongari et al., J. Chem. Theory Comput. 2019, 15, 1, 382–401 <https://doi.org/10.1021/acs.jctc.9b01096>`_

.. [Wilmer2012] `Wilmer et al., J. Phys. Chem. Lett. 2012, 3, 17, 2506–2511 <https://pubs.acs.org/doi/abs/10.1021/jz3008485>`_
