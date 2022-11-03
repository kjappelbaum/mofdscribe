# -*- coding: utf-8 -*-
"""In-dataset predictions for the logarithmitic CO2 Henry coefficients"""
from typing import Optional

import numpy as np

from mofdscribe.datasets import CoREDataset
from mofdscribe.splitters.splitters import DensitySplitter, HashSplitter

from .mofbench import MOFBenchRegression

__all__ = ["LogkHCO2IDBench", "LogkHCO2OODBench"]


class LogkHCO2IDBench(MOFBenchRegression):
    """Benchmarking models for the logarithmic CO2 Henry coefficient under in-domain conditions.

    In-distribution implies that we use a cluster stratified splitter
    that ensures that the ratios of different clusters in the training
    and test set are the same.
    """

    def __init__(
        self,
        model,
        name: str,
        version: Optional[str] = "v0.0.1",
        features: Optional[str] = None,
        model_type: Optional[str] = None,
        reference: Optional[str] = None,
        implementation: Optional[str] = None,
        debug: bool = False,
        patch_in_ds: bool = False,
    ):
        """Initialize the  log KH CO2 interpolation benchmark.

        Args:
            model (object): The model to be benchmarked.
                Must implement the `fit` and `predict` methods.
            name (str): The name of the modeling approach.
            version (str, optional): Version of the dataset to use.
                Defaults to "v0.0.1".
            features (str, optional): Description of the features used in the model.
                Defaults to None.
            model_type (str, optional): Model type (e.g. Conv-Net, BERT, XGBoost).
                Defaults to None.
            reference (str, optional): Reference with more details about modeling approach.
                Defaults to None.
            implementation (str, optional): Link to implementation. Defaults to None.
            debug (bool): If True, use a small dataset (1% of full dataset) for debugging.
                Defaults to False.
            patch_in_ds (bool): If True, the dataset will be patched into the model class
                under the `ds` attribute.
        """
        super().__init__(
            model,
            ds=CoREDataset(version),
            splitter=HashSplitter(
                CoREDataset(version),
                stratification_col="outputs.logKH_CO2",
                sample_frac=0.01 if debug else 1.0,
            ),
            target=["outputs.logKH_CO2"],
            task="logKH_CO2_id",
            k=5,
            version=version,
            features=features,
            name=name,
            model_type=model_type,
            reference=reference,
            implementation=implementation,
            debug=debug,
            patch_in_ds=patch_in_ds,
        )


class LogkHCO2OODBench(MOFBenchRegression):
    """Benchmarking models for the logarithmic CO2 Henry coefficient under "out-of-domain" conditions.

    "Out-of-domain" conditions means that every of the 5 training fold will only see 4 out of the 5
    quantile bins.
    This implies that 2 runs are extrapolative and the other 3 need to "fill holes in the distribution".
    """

    def __init__(
        self,
        model,
        version: Optional[str] = "v0.0.1",
        features: Optional[str] = None,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        reference: Optional[str] = None,
        implementation: Optional[str] = None,
        debug: bool = False,
        patch_in_ds: bool = False,
    ):
        """Initialize the  log KH CO2 extrapolation benchmark.

        Args:
            model (object): The model to be benchmarked.
                Must implement the `fit` and `predict` methods.
            name (str): The name of the modeling approach.
            version (str, optional): Version of the dataset to use.
                Defaults to "v0.0.1".
            features (str, optional): Description of the features used in the model.
                Defaults to None.
            model_type (str, optional): Model type (e.g. Conv-Net, BERT, XGBoost).
                Defaults to None.
            reference (str, optional): Reference with more details about modeling approach.
                Defaults to None.
            implementation (str, optional): Link to implementation. Defaults to None.
            debug (bool): If True, use a small dataset (1% of full dataset) for debugging.
                Defaults to False.
            patch_in_ds (bool): If True, the dataset will be patched into the model class
                under the `ds` attribute.
        """
        super().__init__(
            model,
            ds=CoREDataset(version),
            splitter=DensitySplitter(
                CoREDataset(version),
                sample_frac=0.01 if debug else 1.0,
                density_q=np.linspace(0, 1, 6),
            ),
            target=["outputs.logKH_CO2"],
            task="logKH_CO2_ood",
            k=5,
            version=version,
            features=features,
            name=name,
            model_type=model_type,
            reference=reference,
            implementation=implementation,
            debug=debug,
            patch_in_ds=patch_in_ds,
        )
