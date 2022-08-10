Prediction of the methane deliverable capacity (in-dataset)
============================================================================

Task description
-----------------

MOFs might find use for gas storage, e.g. of methane. The relevant performance indicator is known as the deliverable capacity, which is the difference between the loading at the (high) pressure at which the material is charged and the loading at the (low) pressure at which the material is discharged.


Overview
------------

.. raw:: html
   :file: ch4dc_id_plot_latest.html


Leaderboard
-------------

.. needtable::
   :types: regressionmetrics
   :style: datatables
   :filter: task == "BenchTaskEnum.ch4dc_id"
   :columns: id, name, mean_squared_error, mean_absolute_error, r2_score, max_error, top_50_in_top_50, top_100_in_top_100


Models
------

.. toctree::
   :glob:
   :maxdepth: 1

   ch4dc_id/latest/*
