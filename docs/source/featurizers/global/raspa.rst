
Energy grid histogram
.............................................

In molecular simulations, the interactions between atoms are described using
intermolecular potentials such as the Lennard-Jones potential. Since this
potential must be frequently evaluated one can save computational cost by
pre-computing the potentials on a grid. The grids themselves are not necessarily
fixed-length, and typically high-dimensional. Therefore, they are not directly
used as descriptors. Again, one can solve this problem by "summarizing" the grid
in form of a histogram.


.. figure:: figures/energygrid.svg
  :width: 500
  :alt: Energy grid histogram.

  Converting MOF structures into energy grids and using them as descriptors in
  form of histograms. Figure taken from [Bucior2019_].
