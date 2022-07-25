"""In-dataset predictions for the logarithmitic CO2 Henry coefficients"""
from typing import Optional

from mofdscribe.datasets import CoREDataset
from mofdscribe.splitters import ClusterStratifiedSplitter, DensitySplitter

from .mofbench import MOFBenchRegression

_FEATURES = [
    "total_POV_gravimetric",
    "mc_CRY-chi-0-all",
    "mc_CRY-chi-1-all",
    "mc_CRY-chi-2-all",
    "mc_CRY-chi-3-all",
    "mc_CRY-Z-0-all",
    "mc_CRY-Z-1-all",
    "mc_CRY-Z-2-all",
    "mc_CRY-Z-3-all",
    "mc_CRY-I-0-all",
    "mc_CRY-I-1-all",
    "mc_CRY-I-2-all",
    "mc_CRY-I-3-all",
    "mc_CRY-T-0-all",
    "mc_CRY-T-1-all",
    "mc_CRY-T-2-all",
    "mc_CRY-T-3-all",
    "mc_CRY-S-0-all",
    "mc_CRY-S-1-all",
    "mc_CRY-S-2-all",
    "mc_CRY-S-3-all",
    "D_mc_CRY-chi-0-all",
    "D_mc_CRY-chi-1-all",
    "D_mc_CRY-chi-2-all",
    "D_mc_CRY-chi-3-all",
    "D_mc_CRY-Z-0-all",
    "D_mc_CRY-Z-1-all",
    "D_mc_CRY-Z-2-all",
    "D_mc_CRY-Z-3-all",
    "D_mc_CRY-I-0-all",
    "D_mc_CRY-I-1-all",
    "D_mc_CRY-I-2-all",
    "D_mc_CRY-I-3-all",
    "D_mc_CRY-T-0-all",
    "D_mc_CRY-T-1-all",
    "D_mc_CRY-T-2-all",
    "D_mc_CRY-T-3-all",
    "D_mc_CRY-S-0-all",
    "D_mc_CRY-S-1-all",
    "D_mc_CRY-S-2-all",
    "D_mc_CRY-S-3-all",
    "sum-mc_CRY-chi-0-all",
    "sum-mc_CRY-chi-1-all",
    "sum-mc_CRY-chi-2-all",
    "sum-mc_CRY-chi-3-all",
    "sum-mc_CRY-Z-0-all",
    "sum-mc_CRY-Z-1-all",
    "sum-mc_CRY-Z-2-all",
    "sum-mc_CRY-Z-3-all",
    "sum-mc_CRY-I-0-all",
    "sum-mc_CRY-I-1-all",
    "sum-mc_CRY-I-2-all",
    "sum-mc_CRY-I-3-all",
    "sum-mc_CRY-T-0-all",
    "sum-mc_CRY-T-1-all",
    "sum-mc_CRY-T-2-all",
    "sum-mc_CRY-T-3-all",
    "sum-mc_CRY-S-0-all",
    "sum-mc_CRY-S-1-all",
    "sum-mc_CRY-S-2-all",
    "sum-mc_CRY-S-3-all",
    "sum-D_mc_CRY-chi-0-all",
    "sum-D_mc_CRY-chi-1-all",
    "sum-D_mc_CRY-chi-2-all",
    "sum-D_mc_CRY-chi-3-all",
    "sum-D_mc_CRY-Z-0-all",
    "sum-D_mc_CRY-Z-1-all",
    "sum-D_mc_CRY-Z-2-all",
    "sum-D_mc_CRY-Z-3-all",
    "sum-D_mc_CRY-I-0-all",
    "sum-D_mc_CRY-I-1-all",
    "sum-D_mc_CRY-I-2-all",
    "sum-D_mc_CRY-I-3-all",
    "sum-D_mc_CRY-T-0-all",
    "sum-D_mc_CRY-T-1-all",
    "sum-D_mc_CRY-T-2-all",
    "sum-D_mc_CRY-T-3-all",
    "sum-D_mc_CRY-S-0-all",
    "sum-D_mc_CRY-S-1-all",
    "sum-D_mc_CRY-S-2-all",
    "sum-D_mc_CRY-S-3-all",
    "D_lc-chi-0-all",
    "D_lc-chi-1-all",
    "D_lc-chi-2-all",
    "D_lc-chi-3-all",
    "D_lc-Z-0-all",
    "D_lc-Z-1-all",
    "D_lc-Z-2-all",
    "D_lc-Z-3-all",
    "D_lc-I-0-all",
    "D_lc-I-1-all",
    "D_lc-I-2-all",
    "D_lc-I-3-all",
    "D_lc-T-0-all",
    "D_lc-T-1-all",
    "D_lc-T-2-all",
    "D_lc-T-3-all",
    "D_lc-S-0-all",
    "D_lc-S-1-all",
    "D_lc-S-2-all",
    "D_lc-S-3-all",
    "D_lc-alpha-0-all",
    "D_lc-alpha-1-all",
    "D_lc-alpha-2-all",
    "D_lc-alpha-3-all",
    "D_func-chi-0-all",
    "D_func-chi-1-all",
    "D_func-chi-2-all",
    "D_func-chi-3-all",
    "D_func-Z-0-all",
    "D_func-Z-1-all",
    "D_func-Z-2-all",
    "D_func-Z-3-all",
    "D_func-I-0-all",
    "D_func-I-1-all",
    "D_func-I-2-all",
    "D_func-I-3-all",
    "D_func-T-0-all",
    "D_func-T-1-all",
    "D_func-T-2-all",
    "D_func-T-3-all",
    "D_func-S-0-all",
    "D_func-S-1-all",
    "D_func-S-2-all",
    "D_func-S-3-all",
    "D_func-alpha-0-all",
    "D_func-alpha-1-all",
    "D_func-alpha-2-all",
    "D_func-alpha-3-all",
    "sum-D_lc-chi-0-all",
    "sum-D_lc-chi-1-all",
    "sum-D_lc-chi-2-all",
    "sum-D_lc-chi-3-all",
    "sum-D_lc-Z-0-all",
    "sum-D_lc-Z-1-all",
    "sum-D_lc-Z-2-all",
    "sum-D_lc-Z-3-all",
    "sum-D_lc-I-0-all",
    "sum-D_lc-I-1-all",
    "sum-D_lc-I-2-all",
    "sum-D_lc-I-3-all",
    "sum-D_lc-T-0-all",
    "sum-D_lc-T-1-all",
    "sum-D_lc-T-2-all",
    "sum-D_lc-T-3-all",
    "sum-D_lc-S-0-all",
    "sum-D_lc-S-1-all",
    "sum-D_lc-S-2-all",
    "sum-D_lc-S-3-all",
    "sum-D_lc-alpha-0-all",
    "sum-D_lc-alpha-1-all",
    "sum-D_lc-alpha-2-all",
    "sum-D_lc-alpha-3-all",
    "sum-D_func-chi-0-all",
    "sum-D_func-chi-1-all",
    "sum-D_func-chi-2-all",
    "sum-D_func-chi-3-all",
    "sum-D_func-Z-0-all",
    "sum-D_func-Z-1-all",
    "sum-D_func-Z-2-all",
    "sum-D_func-Z-3-all",
    "sum-D_func-I-0-all",
    "sum-D_func-I-1-all",
    "sum-D_func-I-2-all",
    "sum-D_func-I-3-all",
    "sum-D_func-T-0-all",
    "sum-D_func-T-1-all",
    "sum-D_func-T-2-all",
    "sum-D_func-T-3-all",
    "sum-D_func-S-0-all",
    "sum-D_func-S-1-all",
    "sum-D_func-S-2-all",
    "sum-D_func-S-3-all",
    "sum-D_func-alpha-0-all",
    "sum-D_func-alpha-1-all",
    "sum-D_func-alpha-2-all",
    "sum-D_func-alpha-3-all",
]

__all__ = ("LogkHCO2InterpolationBench",)


class LogkHCO2InterpolationBench(MOFBenchRegression):
    """Benchmarking models for the logarithmic CO2 Henry coefficient under in-distribution conditions.

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
        """
        super().__init__(
            model,
            ds=CoREDataset(version),
            splitter=ClusterStratifiedSplitter(_FEATURES),
            target=["logKH_CO2"],
            task="logKH_CO2_int",
            k=5,
            version=version,
            features=features,
            name=name,
            model_type=model_type,
            reference=reference,
            implementation=implementation,
        )


class LogkHCO2ExtrapolationBench(MOFBenchRegression):
    """Benchmarking models for the logarithmic CO2 Henry coefficient under out-of-distribution conditions."""

    def __init__(
        self,
        model,
        version: Optional[str] = "v0.0.1",
        features: Optional[str] = None,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        reference: Optional[str] = None,
        implementation: Optional[str] = None,
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
        """
        super().__init__(
            model,
            ds=CoREDataset(version),
            splitter=DensitySplitter(),
            target=["logKH_CO2"],
            task="logKH_CO2_ext",
            k=5,
            version=version,
            features=features,
            name=name,
            model_type=model_type,
            reference=reference,
            implementation=implementation,
        )
