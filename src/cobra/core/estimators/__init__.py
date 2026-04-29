"""
Estimator package for COBRA consensus expert pool.

This package provides a unified interface for all estimators used in the
COBRA-style ensemble framework. Each estimator acts as an expert that
generates candidate predictions before distance computation and
aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Estimators form the expert pool of the COBRA architecture. Each model:

- is trained independently on the dataset
- produces predictions for all input samples
- contributes to a consensus prediction through aggregation

This design enables:

- heterogeneous model ensembles
- flexible model selection via factories
- scalable experimentation across model families
- consistent integration with kernel-based weighting

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseEstimator``
    Abstract interface defining ``fit`` and ``predict``.

Factory system
^^^^^^^^^^^^^^

- ``EstimatorFactory``
    Registry-based factory for dynamic estimator creation.

Built-in estimators
^^^^^^^^^^^^^^^^^^^

- ``MeanRegressor``
    Baseline model predicting constant mean value.

- ``LinearRegressorEstimator``
    Ordinary least squares linear regression.

- ``RidgeRegressorEstimator``
    L2-regularized linear regression.

- ``LassoRegressorEstimator``
    L1-regularized regression with sparsity.

- ``KNNRegressorEstimator``
    k-nearest neighbors regression model.

- ``RandomForestRegressorEstimator``
    Ensemble of decision trees.

- ``SVMRegressorEstimator``
    Support vector regression (RBF kernel).

- ``DecisionTreeRegressorEstimator``
    Single decision tree model.

- ``GradientBoostingRegressorEstimator``
    Boosted ensemble of weak learners.

Examples
--------
>>> from cobra.core.estimators import EstimatorFactory

>>> model = EstimatorFactory.create("random_forest")

>>> model.fit(X_train, y_train)

>>> preds = model.predict(X_test)

Exports
-------
All estimator components are exposed for convenient import and
dynamic configuration in pipeline systems.
"""

from .base import (
    BaseEstimator,
    EstimatorFactory,
)

from .builtin import (
    MeanRegressor,
    LinearRegressorEstimator,
    RidgeRegressorEstimator,
    LassoRegressorEstimator,
    KNNRegressorEstimator,
    RandomForestRegressorEstimator,
    SVMRegressorEstimator,
    DecisionTreeRegressorEstimator,
    GradientBoostingRegressorEstimator,
)

__all__ = [
    "BaseEstimator",
    "EstimatorFactory",
    "MeanRegressor",
    "LinearRegressorEstimator",
    "RidgeRegressorEstimator",
    "LassoRegressorEstimator",
    "KNNRegressorEstimator",
    "RandomForestRegressorEstimator",
    "SVMRegressorEstimator",
    "DecisionTreeRegressorEstimator",
    "GradientBoostingRegressorEstimator",
]
