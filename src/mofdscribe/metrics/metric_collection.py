""""Data objects for holding collections of metrics (and performing basic aggregations)."""
from pydantic import BaseModel
from .regression import RegressionMetrics
from typing import List
import numpy as np


class RegressionMetricCollection(BaseModel):
    """Model for regression metric collection."""

    regression_metrics: List[RegressionMetrics]

    def average_metrics(self) -> RegressionMetrics:
        """Return mean of metrics using numpy"""
        return RegressionMetrics(
            mean_squared_error=np.mean(
                [metric.mean_squared_error for metric in self.regression_metrics]
            ),
            mean_absolute_error=np.mean(
                [metric.mean_absolute_error for metric in self.regression_metrics]
            ),
            r2_score=np.mean([metric.r2_score for metric in self.regression_metrics]),
            max_error=np.mean([metric.max_error for metric in self.regression_metrics]),
            mean_absolute_percentage_error=np.mean(
                [metric.mean_absolute_percentage_error for metric in self.regression_metrics]
            ),
            mean_squared_log_error=np.mean(
                [metric.mean_squared_log_error for metric in self.regression_metrics]
            ),
            top_100_in_top_100=np.mean(
                [metric.top_100_in_top_100 for metric in self.regression_metrics]
            ),
            top_500_in_top_500=np.mean(
                [metric.top_500_in_top_500 for metric in self.regression_metrics]
            ),
            top_50_in_top_50=np.mean(
                [metric.top_50_in_top_50 for metric in self.regression_metrics]
            ),
            top_10_in_top_10=np.mean(
                [metric.top_10_in_top_10 for metric in self.regression_metrics]
            ),
            top_5_in_top_5=np.mean([metric.top_5_in_top_5 for metric in self.regression_metrics]),
        )

    def std_metrics(self) -> RegressionMetrics:
        """Compute standard deviation of metrics using numpy."""
        return RegressionMetrics(
            mean_squared_error=np.std(
                [metric.mean_squared_error for metric in self.regression_metrics]
            ),
            mean_absolute_error=np.std(
                [metric.mean_absolute_error for metric in self.regression_metrics]
            ),
            r2_score=np.std([metric.r2_score for metric in self.regression_metrics]),
            max_error=np.std([metric.max_error for metric in self.regression_metrics]),
            mean_absolute_percentage_error=np.std(
                [metric.mean_absolute_percentage_error for metric in self.regression_metrics]
            ),
            mean_squared_log_error=np.std(
                [metric.mean_squared_log_error for metric in self.regression_metrics]
            ),
            top_100_in_top_100=np.std(
                [metric.top_100_in_top_100 for metric in self.regression_metrics]
            ),
            top_500_in_top_500=np.std(
                [metric.top_500_in_top_500 for metric in self.regression_metrics]
            ),
            top_50_in_top_50=np.std(
                [metric.top_50_in_top_50 for metric in self.regression_metrics]
            ),
            top_10_in_top_10=np.std(
                [metric.top_10_in_top_10 for metric in self.regression_metrics]
            ),
            top_5_in_top_5=np.std([metric.top_5_in_top_5 for metric in self.regression_metrics]),
        )
