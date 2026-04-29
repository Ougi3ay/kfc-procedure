"""
CombineClassifier

A COBRA-style ensemble classifier that combines multiple base estimators
using distance-based consensus and kernel-weighted aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Overview
--------
CombineClassifier builds an ensemble of heterogeneous base estimators,
then uses their prediction space to compute similarity between samples.
Final predictions are obtained through kernel-weighted aggregation
over neighbor predictions.

Core idea
---------
Instead of relying on a single model, CombineClassifier:

1. trains multiple base estimators (expert pool)
2. collects their prediction matrix
3. computes distances in prediction space
4. transforms distances into weights via a kernel
5. aggregates weighted neighbor outputs into final prediction

This implements a COBRA-style consensus mechanism.

Design goals
------------
- model-agnostic ensemble construction
- flexible estimator injection (string or object)
- pluggable distance, kernel, and aggregation strategies
- support for heterogeneous model pools
- robust fallback to global majority class
- sklearn-compatible API (fit/predict)

Main components
---------------

Estimators
^^^^^^^^^^
Base learners used as ensemble members.
Examples: logistic regression, random forest, SVM, KNN.

Distance
^^^^^^^^
Measures similarity between prediction vectors.

Kernel
^^^^^^
Transforms distances into weights (indicator, RBF, Laplace, etc.).

Aggregator
^^^^^^^^^^
Combines weighted predictions into final output.

Behavior summary
---------------
- During ``fit``:
    - estimators are trained
    - prediction matrix is stored
    - global majority class is computed
    - core components are initialized

- During ``predict``:
    - prediction matrix is recomputed for test data
    - distances are computed in prediction space
    - kernel produces similarity weights
    - aggregator computes final label per sample

Fallback strategy
-----------------
If no valid neighbors are found for a sample:
→ returns global majority class

Examples
--------
>>> model = CombineClassifier(
...     estimators=["svm", "random_forest"],
...     kernel="rbf",
...     distance="hamming",
...     aggregator="majority_vote"
... )

>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.utils import check_array

from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel, KernelFactory


class CombineClassifier(ABC, SkBaseEstimator):
    """
    COBRA-style ensemble classifier.

    Parameters
    ----------
    estimators : list[str | BaseEstimator] | None
        Base learners used in the ensemble.

    estimators_params : dict[str, Any] | None
        Parameter dictionary per estimator.

    distance : str
        Distance metric in prediction space.

    kernel : str
        Kernel function to convert distances into weights.

    aggregator : str
        Aggregation rule for weighted predictions.

    random_state : int | None
        Random seed (reserved for reproducibility).
    """

    def __init__(
        self,
        estimators: List[Union[str, BaseEstimator]] | None = None,
        estimators_params: Dict[str, Any] | None = None,
        distance: str = "hamming",
        distance_params: Dict[str, Any] | None = None,
        kernel: str = "indicator",
        kernel_params: Dict[str, Any] | None = None,
        aggregator: str = "majority_vote",
        aggregator_params: Dict[str, Any] | None = None,
        random_state: int | None = None,
    ):
        self.estimators = estimators
        self.estimators_params = estimators_params
        self.distance = distance
        self.distance_params = distance_params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.aggregator = aggregator
        self.aggregator_params = aggregator_params
        self.random_state = random_state

    def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):
        """
        Train base estimators.

        Returns
        -------
        list[BaseEstimator]
            Fitted estimator pool.
        """
        default_estimators = [
            "logistic_regression",
            "random_forest",
            "svm",
            "knn",
        ]

        estimators = self.estimators or default_estimators
        machines = []

        for est in estimators:
            if isinstance(est, str):
                params = (self.estimators_params or {}).get(est, {})
                model = EstimatorFactory.create(est, **params)
            elif isinstance(est, BaseEstimator):
                model = est
            else:
                raise ValueError(
                    f"Invalid estimator: {type(est)}. "
                    f"Expected str or BaseEstimator. "
                    f"Available: {EstimatorFactory.available()}"
                )

            model.fit(X_k, y_k)
            machines.append(model)

        return machines

    def _prediction_matrix(self, X: np.ndarray):
        """
        Construct prediction matrix from estimator pool.

        Returns
        -------
        np.ndarray
            Shape: (n_samples, n_estimators)
        """
        cols = []
        for est in self.estimators_:
            preds = est.predict(X)
            cols.append(preds)
        return np.column_stack(cols)

    def _resolve_components(self):
        """Initialize distance, kernel, and aggregator components."""
        self.distance_: BaseDistance = DistanceFactory.create(
            self.distance,
            **(self.distance_params or {}),
        )

        self.kernel_: BaseKernel = KernelFactory.create(
            self.kernel,
            **(self.kernel_params or {}),
        )

        agg_params = self.aggregator_params or {}
        agg_params["classes"] = self.classes_
        self.aggregator_: BaseAggregator = AggregatorFactory.create(
            self.aggregator,
            **agg_params,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ensemble classifier and build consensus space.

        This method:

        1. Fits all base estimators to training data
        2. Generates training prediction matrix
        3. Computes majority class as fallback prediction
        4. Initializes distance, kernel, and aggregator components

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).

        y : np.ndarray
            Target class labels.

        Returns
        -------
        self : CombineClassifier
            Fitted classifier instance (returns self).

        Examples
        --------
        >>> clf = CombineClassifier(estimators=["svm", "random_forest"])
        >>> clf.fit(X_train, y_train)
        """
        self.classes_ = np.unique(y)

        self.estimators_ = self._fit_estimators(X, y)
        self.y_ = self._prediction_matrix(X)

        classes, counts = np.unique(self.y_, return_counts=True)
        self.global_majority_class_ = classes[np.argmax(counts)]

        self._resolve_components()
        return self

    def predict(self, X):
        """
        Predict class labels using kernel-weighted consensus aggregation.

        For each test sample:

        1. Collect all base estimator predictions (prediction matrix)
        2. Compute distance to training predictions in prediction space
        3. Apply kernel to convert distances to similarity weights
        4. Aggregate neighbor training labels using weights

        Parameters
        ----------
        X : np.ndarray
            Test features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).

        Notes
        -----
        When no valid neighbors (zero weights) are found for a sample,
        the method falls back to predicting the global majority class
        observed during training.

        Examples
        --------
        >>> y_pred = clf.predict(X_test)
        """
        X = check_array(X)

        preds = self._prediction_matrix(X)
        outputs = []

        D = self.distance_.matrix(preds, self.y_)
        K = self.kernel_(D)

        for i in range(K.shape[0]):
            w = K[i]
            mask = w > 0

            if not np.any(mask):
                outputs.append(self.global_majority_class_)
                continue

            y_sub = self.y_[mask]
            w_sub = w[mask]

            outputs.append(
                self.aggregator_.aggregate(y_sub, w_sub)
            )

        return np.asarray(outputs)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using kernel-weighted consensus aggregation.

        This method is similar to `predict()`, but instead of returning
        class labels, it returns probability distributions over classes.

        For each test sample:

        1. Collect all base estimator predictions (prediction matrix)
        2. Compute distance to training predictions in prediction space
        3. Apply kernel to convert distances to similarity weights
        4. Aggregate neighbor training labels into probability distribution

        Parameters
        ----------
        X : np.ndarray
            Test features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes).

        Notes
        -----
        When no valid neighbors (zero weights) are found for a sample,
        the method falls back to predicting a one-hot distribution for the
        global majority class observed during training.

        Examples
        --------
        >>> y_prob = clf.predict_prob(X_test)
        """
        X = check_array(X)

        preds = self._prediction_matrix(X)
        outputs = []

        D = self.distance_.matrix(preds, self.y_)
        K = self.kernel_(D)

        for i in range(K.shape[0]):
            w = K[i]
            mask = w > 0

            if not np.any(mask):
                prob_dist = np.zeros(len(self.classes_))
                idx = np.where(self.classes_ == self.global_majority_class_)[0][0]
                prob_dist[idx] = 1.0
                outputs.append(prob_dist)
                continue

            y_sub = self.y_[mask]
            w_sub = w[mask]

            prob_dist = self.aggregator_.aggregate(y_sub, w_sub, return_proba=True)
            outputs.append(prob_dist)

        return np.asarray(outputs)

