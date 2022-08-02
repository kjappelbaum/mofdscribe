# -*- coding: utf-8 -*-
"""In-dataset predictions for PBE bandgaps."""
from typing import Optional

from mofdscribe.datasets import QMOFDataset
from mofdscribe.splitters.splitters import HashSplitter

from .mofbench import MOFBenchRegression

__all__ = ("PBEBandGapBench",)


class PBEBandGapBench(MOFBenchRegression):
    """Benchmarking models for the PBE bandgap under in-distribution conditions.

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
    ):
        """Initialize the PBE bandgap benchmark.

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
        """
        super().__init__(
            model,
            ds=QMOFDataset(version),
            splitter=HashSplitter(
                QMOFDataset(version),
                stratification_col="outputs.pbe.bandgap",
                sample_frac=0.01 if debug else 1.0,
            ),
            target=["logKH_CO2"],
            task="pbe_bandgap",
            k=5,
            version=version,
            features=features,
            name=name,
            model_type=model_type,
            reference=reference,
            implementation=implementation,
            debug=debug,
        )