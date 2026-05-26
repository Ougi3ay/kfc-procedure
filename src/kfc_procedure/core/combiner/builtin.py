"""
Regression combiner strategies.

These combiners merge divergence-specific predictions into a final
ensemble prediction during the C-step.

Supported strategies
--------------------

"mean"          – row-wise mean (stateless)
"weighted_mean" – OLS-learned weighted average
"stacking"      – meta-regressor trained on prediction matrix
"""

from __future__ import annotations
from collections import Counter

import numpy as np
from abc import ABC

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, LogisticRegression

from cobra.combine_classifier import CombineClassifier
from cobra.gradientcobra import GradientCOBRA
from cobra.mixcobra import MixCOBRARegressor
from kfc_procedure.core.combiner import BaseCombiner
from kfc_procedure.core.combiner.base import CombinerFactory

@CombinerFactory.register("mean", categories={"regression"})
class MeanCombiner(BaseCombiner):
    """
    Simple row-wise mean combiner.

    No training required.
    """

    def fit(self, X: np.ndarray, y=None):
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        return np.mean(X, axis=1)

@CombinerFactory.register("weighted_mean", categories={"regression"})
class WeightedMeanCombiner(BaseCombiner):
    """
    Learns optimal linear weights using Ordinary Least Squares.

    Model:
        y ≈ Xw

    where:
        X = prediction matrix (n_samples, n_models)
        w = learned weights
    """

    def __init__(self, fit_intercept: bool = False):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        self.model.fit(X, y)
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        return self.model.predict(X)


@CombinerFactory.register("stacking_regressor", categories={"regression"})
class StackingCombiner(BaseCombiner):
    """
    Meta-learning combiner using a regression model.

    Learns:
        f(X) -> y

    where:
        X = predictions from base models
        y = ground truth
    """

    def __init__(self, meta_model=None):
        self.meta_model = meta_model or LinearRegression()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(X, y)

        self._is_fitted = True
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("StackingCombiner is not fitted.")

        X = np.asarray(X)
        return self.meta_model_.predict(X)

@CombinerFactory.register("gradientcobra", categories={"regression"})
class GradientCOBRACombiner(BaseCombiner):
    def __init__(
        self,
        **cobra_params
    ):
        self.cobra = GradientCOBRA(**cobra_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.cobra.fit(X, y)
        return self
    
    def combine(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict(X)

@CombinerFactory.register("mixcobra", categories={"regression"})
class GradientCOBRACombiner(BaseCombiner):
    def __init__(
        self,
        **cobra_params
    ):
        self.cobra = MixCOBRARegressor(**cobra_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.cobra.fit(X, y)
        return self
    
    def combine(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict(X)
    

@CombinerFactory.register("majority_vote", categories={"classification"})
class MajorityVoteCombiner(BaseCombiner):
    """
    Row-wise majority vote (mode) across base classifier predictions.

    This is a stateless hard-voting ensemble combiner.
    """

    def fit(self, X: np.ndarray, y=None):
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        n_samples = X.shape[0]
        outputs = np.empty(n_samples, dtype=object)

        for i in range(n_samples):
            row = X[i]
            outputs[i] = Counter(row).most_common(1)[0][0]

        return outputs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.combine(X)

@CombinerFactory.register("stacking_classifier", categories={"classification"})
class StackingClassifierCombiner(BaseCombiner):
    """
    Meta-classifier trained on prediction matrix.

    Learns:
        y = f(X)

    where:
        X = predictions from base classifiers
        y = true labels
    """

    def __init__(self, meta_model=None):
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(X, y)

        self._is_fitted = True
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("StackingClassifierCombiner is not fitted.")

        X = np.asarray(X)
        return self.meta_model_.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.combine(X)

@CombinerFactory.register("combiner_classifier", categories={"classification"})
class CobraClassifierCombiner(BaseCombiner):
    def __init__(
        self,
        **cobra_params
    ):
        self.cobra = CombineClassifier(**cobra_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.cobra.fit(X, y)
        return self
    
    def combine(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict_proba(X)
