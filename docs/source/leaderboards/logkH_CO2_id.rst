Prediction of logarithmic Henry coefficient for carbon dioxide (in-dataset)
============================================================================

Task description
-----------------

The Henry coefficient is the slope of the gas adsorption isotherm for :math:`p\to0`.
It can be computed using grand canonical Monte Carlo.
For this task, we use the data computed in [Moosavi2021]_ where the framework is modeled using
the UFF force field [UFF]_ and the carbon dioxide using the TraPPE [Trappe]_ force field.

To estimate the predictive performance of models on in-distribution data, this task uses
a splitter that stratifies on k-means clusters in the dataset and therefore ensures that the
test data is similar to the training data.


Overview
------------

.. raw:: html
   :file: logKH_CO2_id_plot_latest.html


Leaderboard
-------------

.. needtable::
   :types: regressionmetrics
   :style: datatables
   :filter: task == "BenchTaskEnum.logKH_CO2_id"
   :columns: id, name, mean_squared_error, mean_absolute_error, r2_score, max_error, top_50_in_top_50, top_100_in_top_100


Models
------

.. toctree::
   :glob:
   :maxdepth: 1

   logKH_CO2_id_models/latest/*
