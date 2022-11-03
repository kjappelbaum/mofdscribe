# -*- coding: utf-8 -*-
"""In-dataset predictions for the methane deliverable capacity"""
from typing import Optional

from mofdscribe.bench.mofbench import MOFBenchRegression
from mofdscribe.datasets import CoREDataset
from mofdscribe.splitters.splitters import HashSplitter

__all__ = ["CH4DCIDBench"]


class CH4DCIDBench(MOFBenchRegression):
    """Benchmarking models for the methane deliverable capacity under in-domain conditions.

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
        """Initialize the CH4DC interpolation benchmark.

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
                stratification_col="outputs.CH4DC",
                sample_frac=0.01 if debug else 1.0,
            ),
            target=["outputs.CH4DC"],
            task="ch4dc_id",
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
