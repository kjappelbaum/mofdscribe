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
For describing the pore geometry, we heavily rely on methods implemented in the `zeopp <>`_ package.


"Chemistry" descriptors
--------------------------



Topological descriptors
-------------------------
For many applications of porous materials the _shape_ of the material, e.g., the pore shape, is relevant for the application.
Topology is the branch of mathematics that deals with shapes and one of the most widely used topological techniques to describe shapes is known as persistent homology.


Vectorizing persistence diagrams
..................................
For many machine learning models, fixed length vectors are required.  Persistence diagrams, however, are not fixed length. In `mofdscribe` we provide two methods to vectorize persistence diagrams.
