from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from kfc_procedure.cobra.core.factory import BaseFactory


class BaseAggregator(ABC):
    """
    Base class for COBRA aggregation strategies.

    This class defines the interface for all aggregation methods used
    in the COBRA framework. Aggregators combine predictions from multiple
    base estimators using optional weights (e.g., kernel similarities).

    Two execution modes are supported:
    -------------------------------
    1. Single aggregation (aggregate)
    2. Batch aggregation (aggregate_matrix)

    Batch mode is used in:
    - Cross-validation optimization
    - GradientCOBRA acceleration
    - Large-scale prediction pipelines

    Design principle:
    -----------------
    - Fully compatible with vectorized or iterative implementations
    - Enforces consistent input validation across aggregators
    """

    @abstractmethod
    def aggregate(
        self,
        values: np.ndarray,
        weights: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Aggregate a single set of estimator predictions.

        Parameters
        ----------
        values : np.ndarray
            Predictions from base estimators (shape: n_models)

        weights : np.ndarray or None
            Optional weights for each estimator (shape: n_models)

        Returns
        -------
        Any
            Aggregated prediction (scalar or vector depending on task)

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass
        """
        raise NotImplementedError

    def aggregate_matrix(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        fallback: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Batch aggregation over multiple queries.

        This method applies `aggregate()` independently to each query.

        Parameters
        ----------
        values : np.ndarray
            Shared estimator predictions (shape: n_models)

        weights : np.ndarray
            Weight matrix (shape: n_queries × n_models)

        fallback : float, optional
            Value used when aggregation fails or degenerates

        Returns
        -------
        np.ndarray
            Aggregated predictions for each query (shape: n_queries)

        Raises
        ------
        ValueError
            If weight dimensionality or shape is invalid
        """

        V = np.asarray(values)
        W = np.asarray(weights)

        if W.ndim != 2:
            raise ValueError(
                f"weights must be 2D (n_queries, n_models), got {W.shape}"
            )

        out = np.empty(W.shape[0], dtype=object)
        for i in range(W.shape[0]):
            out[i] = self.aggregate(V, W[i], fallback=fallback, **kwargs)

        return np.asarray(out)

    def aggregate_proba(
        self,
        values: np.ndarray,
        weights: np.ndarray | None = None,
        classes: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Aggregate probabilistic outputs for classification tasks.

        This method must be implemented by classification aggregators.

        Parameters
        ----------
        values : np.ndarray
            Probability matrix from base classifiers

        weights : np.ndarray or None
            Optional weights for each estimator

        classes : np.ndarray or None
            Class labels (if required by implementation)

        Returns
        -------
        np.ndarray
            Aggregated class probabilities

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass
        """
        raise NotImplementedError

class AggregatorFactory(BaseFactory):
    pass