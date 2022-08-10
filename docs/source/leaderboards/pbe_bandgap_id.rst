Prediction of the PBE bangap (in-dataset)
============================================================================

Task description
-----------------

The electronic band gaps are a relevant indicator for applications such as electro- or photocatalysis, sensing, energy storage. 
Rosen et al. [Rosen2021]_ [Rosen2022]_  performed a large-scale screening in which they computed the band gaps of metal organic frameworks using DFT, including with the PBE functional. 

Overview
------------

.. raw:: html
   :file: pbe_bandgap_id_plot_latest.html


Leaderboard
-------------

.. needtable::
   :types: regressionmetrics
   :style: datatables
   :filter: task == "BenchTaskEnum.pbe_bandgap_id"
   :columns: id, name, mean_squared_error, mean_absolute_error, r2_score, max_error, top_50_in_top_50, top_100_in_top_100


Models
------

.. toctree::
   :glob:
   :maxdepth: 1

   pbe_bandgap_id_models/latest/*
