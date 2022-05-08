import numpy as np


def generalized_degree_of_freedom(
    model, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-5, mc_samples: int = 10
) -> float:
    """Calculate the generalized degree of freedom of a model."""
    # [Computing AIC for black-box models using Generalised Degrees of Freedom:  a comparison with cross-validation](https://arxiv.org/pdf/1603.02743)
    # [Degrees of Freedom in Deep Neural Networks](https://arxiv.org/pdf/1603.09260)
