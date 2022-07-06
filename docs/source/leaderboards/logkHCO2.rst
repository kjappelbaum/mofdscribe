Prediction of logarithmic Henry coefficient for carbon dioxide
==================================================================

Task description
-----------------

The Henry coefficient is the slope of the gas adsorption isotherm for :math:`p\to0`. 
It can be computed using grand canonical Monte Carlo. 
For this task, we use the data computed in [Moosavi2021]_ where the framework is modeled using 
the UFF force field [UFF]_ and the carbon dioxide using the TraPPE [Trappe]_ force field.

To estimate the predictive performance of models on in-distribution data, this task uses 
a splitter that stratifies on k-means clusters in the dataset and therefore ensures that the 
test data is similar to the training data.


Leaderboard
-------------

.. needtable::
   :types: regressionmetrics
   :style: datatable
   :columns: id, modelname, mae, mse





Models
------

.. toctree::
   :maxdepth: 2

   logKHCO2_models/dummy_mean