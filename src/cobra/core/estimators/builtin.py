"""
Estimator wrappers for COBRA-style expert pools.

This module provides a unified estimator layer used in COBRA-based
ensemble systems such as GradientCOBRA and MixCOBRA.

It standardizes scikit-learn models under a single interface so they
can be:

- registered via a factory
- swapped dynamically in pipelines
- used consistently in ensemble expert pools

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Design goals
------------
- unify estimator interface across models
- support factory-based instantiation
- reduce boilerplate for sklearn models
- enable hyperparameter injection via constructors
- ensure compatibility with ensemble aggregation systems
"""

from __future__ import annotations

from typing import Optional
from numpy.typing import ArrayLike

import numpy as np

from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .base import BaseEstimator, EstimatorFactory


class SklearnEstimator(BaseEstimator):
    """
    Generic wrapper for scikit-learn estimators.

    This class adapts any sklearn-compatible model to the COBRA
    estimator interface.

    It ensures consistent behavior across all expert models used
    in the ensemble pool.

    Parameters
    ----------
    estimator : SkBaseEstimator
        Any scikit-learn compatible estimator instance.

    Notes
    -----
    This wrapper assumes the underlying estimator follows sklearn's
    fit/predict API.

    Examples
    --------
    >>> wrapper = SklearnEstimator(LinearRegression())
    >>> wrapper.fit(X, y)
    >>> preds = wrapper.predict(X_test)
    """

    def __init__(self, estimator: SkBaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, x: ArrayLike, y: ArrayLike) -> "SklearnEstimator":
        """
        Fit the underlying estimator.

        Parameters
        ----------
        x : ArrayLike
            Training features.

        y : ArrayLike
            Target values.

        Returns
        -------
        SklearnEstimator
            Fitted estimator (self).
        """
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        x : ArrayLike
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values as float array.
        """
        return np.asarray(self.estimator.predict(x), dtype=float)

    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        """
        Return class probabilities (if supported).

        Parameters
        ----------
        x : ArrayLike
            Input features.

        Returns
        -------
        np.ndarray
            Probability estimates.

        Raises
        ------
        NotImplementedError
            If estimator does not support probability prediction.
        """
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x), dtype=float)

        raise NotImplementedError(
            f"{self.estimator.__class__.__name__} "
            "does not support predict_proba."
        )


@EstimatorFactory.register("mean_regressor", "dummy_mean")
class MeanRegressor(BaseEstimator):
    """
    Mean baseline regressor.

    Predicts the average value of training targets regardless of input.

    This serves as a simple sanity-check baseline inside expert pools.

    Notes
    -----
    Useful for debugging and baseline comparison.
    """

    def __init__(self) -> None:
        self.estimator = DummyRegressor(strategy="mean")

    def fit(self, x: ArrayLike, y: ArrayLike) -> "MeanRegressor":
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        return np.asarray(self.estimator.predict(x), dtype=float)


@EstimatorFactory.register("linear")
class LinearRegressorEstimator(SklearnEstimator):
    """
    Linear regression estimator.

    Models a linear relationship between input features and target.

    Useful as a fast and interpretable baseline model.
    """

    def __init__(self) -> None:
        super().__init__(LinearRegression())


@EstimatorFactory.register("ridge")
class RidgeRegressorEstimator(SklearnEstimator):
    """
    Ridge regression estimator (L2 regularization).

    Reduces overfitting by penalizing large coefficients.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(Ridge(alpha=alpha))


@EstimatorFactory.register("lasso")
class LassoRegressorEstimator(SklearnEstimator):
    """
    Lasso regression estimator (L1 regularization).

    Encourages sparsity by driving some coefficients to zero,
    effectively performing feature selection.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(Lasso(alpha=alpha))


@EstimatorFactory.register("knn")
class KNNRegressorEstimator(SklearnEstimator):
    """
    K-Nearest Neighbors regressor.

    Non-parametric model that predicts based on local similarity
    in feature space.
    """

    def __init__(self, n_neighbors: int = 7) -> None:
        super().__init__(KNeighborsRegressor(n_neighbors=n_neighbors))


@EstimatorFactory.register("random_forest")
class RandomForestRegressorEstimator(SklearnEstimator):
    """
    Random Forest regressor.

    Ensemble of decision trees trained on bootstrapped samples.

    Captures nonlinear relationships and interactions robustly.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
            )
        )


@EstimatorFactory.register("svm")
class SVMRegressorEstimator(SklearnEstimator):
    """
    Support Vector Regression (SVR).

    Uses kernel methods (RBF kernel by default) to model
    nonlinear relationships in data.
    """

    def __init__(self, C: float = 5.0, epsilon: float = 0.05) -> None:
        super().__init__(
            SVR(C=C, epsilon=epsilon, kernel="rbf")
        )


@EstimatorFactory.register("logistic_regression")
class LogisticRegressionEstimator(SklearnEstimator):
    """
    Logistic regression estimator.

    Primarily a classification model, but used here as a
    probabilistic scoring estimator in ensemble settings.
    """

    def __init__(
        self,
        max_iter: int = 5000,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
            )
        )


@EstimatorFactory.register("decision_tree")
class DecisionTreeRegressorEstimator(SklearnEstimator):
    """
    Decision tree regressor.

    Simple nonlinear model based on hierarchical feature splits.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=random_state,
            )
        )


@EstimatorFactory.register("gradient_boosting")
class GradientBoostingRegressorEstimator(SklearnEstimator):
    """
    Gradient boosting regressor.

    Sequential ensemble of weak learners trained to reduce error.

    Typically provides strong performance on structured data.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )
        )
