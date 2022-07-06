# -*- coding: utf-8 -*-
"""Data objects for holding collections of metrics (and performing basic aggregations)."""
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from .regression import RegressionMetrics, RegressionMetricsConcat


class RegressionMetricCollection(BaseModel):
    """Model for regression metric collection."""

    regression_metrics: List[RegressionMetrics]
    fit_timings: Optional[List[float]]
    inference_timings: Optional[List[float]]

    def concatenated_metrics(self) -> RegressionMetrics:
        """Return concatenated metrics."""
        return RegressionMetricsConcat(
            mean_squared_error=[metric.mean_squared_error for metric in self.regression_metrics],
            mean_absolute_error=[metric.mean_absolute_error for metric in self.regression_metrics],
            r2_score=[metric.r2_score for metric in self.regression_metrics],
            max_error=[metric.max_error for metric in self.regression_metrics],
            mean_absolute_percentage_error=[
                metrics.mean_absolute_percentage_error for metrics in self.regression_metrics
            ],
            top_100_in_top_100=[metric.top_100_in_top_100 for metric in self.regression_metrics],
            top_500_in_top_500=[metric.top_500_in_top_500 for metric in self.regression_metrics],
            top_50_in_top_50=[metric.top_50_in_top_50 for metric in self.regression_metrics],
            top_10_in_top_10=[metric.top_10_in_top_10 for metric in self.regression_metrics],
            top_5_in_top_5=[metric.top_5_in_top_5 for metric in self.regression_metrics],
        )

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

    def average_fit_timings(self) -> float:
        """Return mean of fit timings using numpy."""
        return np.mean(self.fit_timings)

    def std_fit_timings(self) -> float:
        """Return standard deviation of fit timings using numpy."""
        return np.std(self.fit_timings)

    def average_inferences_timings(self) -> float:
        """Return mean of inference timings using numpy."""
        return np.mean(self.inference_timings)

    def std_inference_timings(self) -> float:
        """Return standard deviation of inference timings using numpy."""
        return np.std(self.inference_timings)
