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
from .base import (
    BaseBregmanDivergence,
    BregmanDivergenceFactory
)

from .euclidean import SquaredEuclidean
from .gkl import GKLDivergence
from .itakura_saito import ItakuraSaito
from .logistic import LogisticLoss

__all__ = [
    "BaseBregmanDivergence",
    "BregmanDivergenceFactory",
    "SquaredEuclidean",
    "GKLDivergence",
    "ItakuraSaito",
    "LogisticLoss",
]
