"""
Clustering algorithms and divergence measures.

This package provides clustering methods used throughout the
KFCProcedure framework, including Bregman-divergence-based clustering
algorithms and the divergence functions that support them.

The primary estimator is :class:`BregmanKMeans`, a generalized k-means
algorithm capable of operating with arbitrary Bregman divergences.

Subpackages
-----------
divergences
    Built-in Bregman divergence implementations and divergence
    registration utilities.

Classes
-------
BregmanKMeans
    Lloyd-style clustering algorithm using Bregman divergences.

Available Divergences
---------------------
The following divergences are provided by default:

* Squared Euclidean divergence
* Generalized Kullback-Leibler divergence
* Itakura-Saito divergence
* Logistic divergence

Notes
-----
Bregman divergences generalize classical distance measures and provide
a unified framework for clustering data generated from different
exponential-family distributions, including Gaussian, Poisson,
Gamma, and Bernoulli models.

References
----------
Banerjee, A., Merugu, S., Dhillon, I. S., and Ghosh, J. (2005).
"Clustering with Bregman Divergences."
Journal of Machine Learning Research, 6, 1705-1749.
"""

from .bregman import BregmanKMeans
from . import divergences

__all__ = [
    "BregmanKMeans",
    "divergences"
]