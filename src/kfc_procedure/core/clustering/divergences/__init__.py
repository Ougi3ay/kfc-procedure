"""
Divergences
-------------------------
Bregman divergence abstractions and built-in implementations.

Importing this package registers all four paper divergences:

    "euclidean"  –  Squared Euclidean  /  Gaussian
    "gkl"        –  Generalised KL     /  Poisson
    "is"         –  Itakura-Saito      /  Gamma / spectral
    "logistic"   –  Logistic loss      /  Bernoulli
"""
from kfc_procedure.core.clustering.divergences.base import BregmanDivergence, BregmanDivergenceFactory
from kfc_procedure.core.clustering.divergences.builtin import (
    GKLDivergence,
    ItakuraSaito,
    LogisticLoss,
    SquaredEuclidean,
)

__all__ = [
    "BregmanDivergence",
    "BregmanDivergenceFactory",
    "SquaredEuclidean",
    "GKLDivergence",
    "ItakuraSaito",
    "LogisticLoss",
]
