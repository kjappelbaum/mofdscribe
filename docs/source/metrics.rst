Metrics
===================

In order to compare our models we need to score them using a metric. 
Most commonly used in are scores such as accuracy, precision, recall, or the mean absolute error for regression problem. 

However, these metrics are not always the best choice. 
It is well known, for instance, that accuracy is not a good metric for imbalanced datasets.
However, even beyond such considerations it is important to take into consideration for what purpose the model is used.

For materials discovery, this often implies that a metric that measures how many of the top materials we find is more 
important than an averaged, overall score.

``mofdscribe`` provides some utilities to help with this in the ``mofdscribe.metrics`` subpackage.