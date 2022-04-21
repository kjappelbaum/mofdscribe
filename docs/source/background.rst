Featurizing MOFs
===================

.. list-table:: Overview over implemented featurizers
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Assumes locality
     - Contains chemistry information
     - Contains explicit geometry information
     - Works on structure graph
     - Original reference
   * - Row 1, column 1
     -
     - Row 1, column 3
   * - Row 2, column 1
     - Row 2, column 2
     - Row 2, column 3

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
