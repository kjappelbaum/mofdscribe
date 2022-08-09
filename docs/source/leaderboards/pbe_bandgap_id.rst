Prediction of the PBE bangap (in-dataset)
============================================================================

Task description
-----------------


Overview
------------

.. raw:: html
   :file: pbe_bandgap_id_plot_latest.html


Leaderboard
-------------

.. needtable::
   :types: regressionmetrics
   :style: datatables
   :filter: task == "BenchTaskEnum.pbe_bandgap_id" AND version == "v0.0.1"
   :columns: id, name, mean_squared_error, mean_absolute_error, r2_score, max_error, top_50_in_top_50, top_100_in_top_100


Models
------

.. toctree::
   :glob:
   :maxdepth: 1

   pbe_bandgap_id_models/latest/*
