"""
Regression and classification combiner strategies.

This module defines a collection of combiner methods used to aggregate
predictions from multiple base estimators into a final ensemble output.
These combiners are primarily used in the C-step of COBRA-style and
Bregman-based ensemble learning frameworks.

Each combiner operates on a prediction matrix

    X ∈ R^{n_samples × n_models}

where each column corresponds to predictions from a base model.

Overview
--------
The module supports both regression and classification aggregation
strategies, ranging from simple statistical rules to learned meta-models.

Regression combiners
---------------------
mean
    Simple row-wise arithmetic mean (no training required).

weighted_mean
    Linear regression-based weighted combination of base predictions.

stacking
    Meta-regressor trained on the prediction matrix.

gradientcobra
    Uses GradientCOBRA as a learned nonparametric combiner.

mixcobra
    Uses MixCOBRARegressor for adaptive ensemble aggregation.

Classification combiners
-------------------------
majority_vote
    Hard voting (mode) across base classifier predictions.

stacking_classifier
    Logistic regression-based stacking classifier.

combined_classifier
    Uses COBRA-based CombinedClassifier for probabilistic aggregation.

Mathematical formulation
------------------------
Given base model predictions X = [x_1, ..., x_K], a combiner learns:

    f(X) → y

where f may be:

* a deterministic rule (mean, vote)
* a linear model (weighted mean, stacking)
* a nonparametric ensemble method (COBRA variants)

Notes
-----
All combiners follow a scikit-learn compatible API:

* fit(X, y)
* combine(X)
* predict(X) (where applicable)

Stateless combiners (e.g., mean, majority_vote) do not require fitting.

References
----------
Bühlmann, P. (2012).
"Bagging, Boosting and Ensemble Methods."

Breiman, L. (1996).
"Stacked regressions."

Biau, G., & Fischer, A. (2012).
"Analyzing random forests with nearest-neighbor methods."

Dudoit, S., & van der Laan, M. J. (2003).
"Asymptotics of cross-validated risk estimation in estimator stacking."
"""

from .base import BaseCombiner, CombinerFactory
from . import regression
from . import classification

__all__ = [
    "BaseCombiner",
    "CombinerFactory",
    "regression",
    "classification",
]
