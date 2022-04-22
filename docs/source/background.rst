Featurizing MOFs
===================
.. Potential additional categories: High-dimensional? Models pores? 

.. list-table:: Overview of implemented featurizers
   :widths: 25 20 20 20 20 25 
   :header-rows: 1

   * - Name
     - Assumes locality
     - Contains chemistry information 
     - Derived using geometry information
     - Derived using structure graph
     - Original reference
   * - `RACS <:py:class:`mofdscribe.chemistry.racs.RACS>`_
     - ✅
     - ✅
     - ❌
     - ✅
     - `Moosavi et al., Nature Communications, 2021 <https://www.nature.com/articles/s41467-020-17755-8>`_ 
   * - `APRDF <:py:class:`mofdscribe.chemistry.aprdf.APRDF>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - `Fernandez et al., J. Phys. Chem. C., 2013 <https://pubs.acs.org/doi/full/10.1021/jp404287t>`_
   * - `EnergyGridHistogram <:py:class:`mofdscribe.chemistry.energygrid.EnergyGridHistogram>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - `Bucior et al.,  Mol. Syst. Des. Eng., 2019 <https://pubs.rsc.org/en/content/articlelanding/2019/me/c8me00050f>`_
   * - `PartialChargeStats <:py:class:`mofdscribe.chemistry.partialchargestats.PartialChargeStats>`_
     - ❌
     - ✅
     - ✅
     - ❌
     - 
   * - `PoreDiameters <:py:class:`mofdscribe.pore.geometric_properties.PoreDiameters>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - 
   * - `SurfaceArea <:py:class:`mofdscribe.pore.geometric_properties.SurfaceArea>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - 
   * - `AccessibleVolume <:py:class:`mofdscribe.pore.geometric_properties.AccessibleVolume>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - 
   * - `RayTracingHistogram <:py:class:`mofdscribe.pore.geometric_properties.RayTracingHistogram>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - 
   * - `PoreSizeDistribution <:py:class:`mofdscribe.pore.geometric_properties.PoreSizeDistribution>`_
     - ❌
     - ❌
     - ✅
     - ❌
     - 
   * - `PHImage <:py:class:`mofdscribe.topology.ph_image.PHImage>`_
     - ❌
     - ✅ (optionally)
     - ✅ 
     - ❌
     - 
   * - `PHVect <:py:class:`mofdscribe.topology.ph_vect.PHVect>`_
     - ❌
     - ✅ (optionally)
     - ✅ 
     - ❌
     - 
 

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



"Chemistry" descriptors
--------------------------



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


[Adams2017]_, `Journal of Machine Learning Research 18 (2017) 1-35 <https://jmlr.csail.mit.edu/papers/volume18/16-337/16-337.pdf>.`_
[Perea]_, Jose A. Perea, Elizabeth Munch, Firas A. Khasawneh, Approximating Continuous Functions on Persistence Diagrams Using Template Functions, arXiv:1902.07190
[Tymochko]_, Sarah Tymochko, Elizabeth Munch, Firas A. Khasawneh, Adaptive Partitioning for Template Functions on Persistence Diagrams, arXiv:1910.08506v1
