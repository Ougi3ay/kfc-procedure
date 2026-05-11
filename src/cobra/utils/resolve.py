"""
Factory resolver utilities for COBRA pipeline configuration.

This module provides helper functions that convert string-based
configuration entries into instantiated COBRA components using
their respective factories.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
In COBRA-style systems, components are often defined declaratively
(e.g., in YAML or config dictionaries). These resolvers convert those
declarations into executable objects.

They act as the glue layer between:

- configuration files (strings / dicts)
- factory-based component instantiation
- runtime pipeline construction

Design goals
------------
- unify instantiation logic across components
- support both string and pre-instantiated objects
- enable flexible configuration-driven pipelines
- provide clear error handling for invalid components
- reduce boilerplate in pipeline construction

Supported components
--------------------

This module resolves the following pipeline stages:

- Estimators
- Kernels
- Splitters
- Loss functions
- Distance metrics
- Aggregators

Behavior
--------
Each resolver follows the same pattern:

1. If input is a string → use corresponding Factory.create()
2. If input is already an object → return as-is
3. If input is None → use defaults (where applicable)

Examples
--------
>>> estimator = resolve_from_estimators("ridge", None, ["ridge"])
>>> kernel = resolve_from_kernel("rbf", {"gamma": 1.0})
>>> loss = resolve_from_loss("mse", None)
"""

from __future__ import annotations

from typing import Any, Iterable

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import check_X_y

from cobra.core.aggregators.base import AggregatorFactory
from cobra.core.distances.base import DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import KernelFactory
from cobra.core.losses.base import LossFactory
from cobra.core.splitters.base import BaseDataSplitter, SplitterFactory
from cobra.core.types import SplitIndices, TrainingContext


def resolve_from_estimators(
    estimators: list[Any] | str | None,
    estimators_params: dict[str, Any] | None,
    default_estimators: list[str],
) -> list[Any]:
    """
    Resolve estimator configurations into instantiated objects.

    Parameters
    ----------
    estimators : list[Any] | str | None
        Estimator names or already-instantiated objects.

    estimators_params : dict[str, Any] | None
        Shared parameters passed to estimator constructors.

    default_estimators : list[str]
        Fallback estimator names if none are provided.

    Returns
    -------
    list[Any]
        List of instantiated estimators.

    Raises
    ------
    TypeError
        If estimators is not a valid type.

    ValueError
        If an unknown estimator name is provided.

    Notes
    -----
    This function supports hybrid inputs (mix of strings and objects)
    to allow flexible pipeline construction.
    """
    if estimators is None:
        estimators = default_estimators

    if isinstance(estimators, str):
        estimators = [estimators]

    if not isinstance(estimators, Iterable):
        raise TypeError("estimators must be a list, string, or None")

    resolved: list[Any] = []

    for est in estimators:
        if isinstance(est, str):
            name = est.lower()

            try:
                resolved_est = EstimatorFactory.create(
                    name,
                    **(estimators_params or {}),
                )
            except Exception as e:
                raise ValueError(
                    f"Unknown estimator '{est}'. "
                    f"Available: {EstimatorFactory.available()}"
                ) from e
        else:
            resolved_est = est

        resolved.append(resolved_est)

    return resolved


def resolve_from_kernel(
    kernel: str | Any,
    kernel_params: dict[str, Any] | None,
):
    """
    Resolve kernel configuration into an instantiated kernel.

    Parameters
    ----------
    kernel : str | Any
        Kernel name or instance.

    kernel_params : dict[str, Any] | None
        Kernel constructor parameters.

    Returns
    -------
    Any
        Instantiated kernel.
    """
    return KernelFactory.create(kernel, **(kernel_params or {}))


def resolve_from_splitter(
    splitter: str | Any,
    splitter_params: dict[str, Any] | None,
):
    """
    Resolve splitter configuration into an instantiated splitter.

    Parameters
    ----------
    splitter : str | Any
        Splitter name or instance.

    splitter_params : dict[str, Any] | None
        Splitter parameters.

    Returns
    -------
    Any
        Instantiated splitter.
    """
    return SplitterFactory.create(splitter, **(splitter_params or {}))


def resolve_from_loss(
    loss: str | Any,
    loss_params: dict[str, Any] | None,
):
    """
    Resolve loss configuration into an instantiated loss function.

    Parameters
    ----------
    loss : str | Any
        Loss name or instance.

    loss_params : dict[str, Any] | None
        Loss parameters.

    Returns
    -------
    Any
        Instantiated loss function.
    """
    return LossFactory.create(loss, **(loss_params or {}))


def resolve_from_distance(
    distance: str | Any,
    distance_params: dict[str, Any] | None,
):
    """
    Resolve distance metric configuration into an instantiated object.

    Parameters
    ----------
    distance : str | Any
        Distance metric name or instance.

    distance_params : dict[str, Any] | None
        Distance parameters.

    Returns
    -------
    Any
        Instantiated distance metric.
    """
    return DistanceFactory.create(distance, **(distance_params or {}))


def resolve_from_aggregator(
    aggregator: str | Any,
    aggregator_params: dict[str, Any] | None,
):
    """
    Resolve aggregator configuration into an instantiated object.

    Parameters
    ----------
    aggregator : str | Any
        Aggregator name or instance.

    aggregator_params : dict[str, Any] | None
        Aggregator parameters.

    Returns
    -------
    Any
        Instantiated aggregator.
    """
    return AggregatorFactory.create(aggregator, **(aggregator_params or {}))

def fit_estimators(
    X,
    y,
    estimators_params=None,
    estimators=None,
    n_jobs=1
):
    """
    Fit a pool of estimators in parallel using EstimatorFactory.

    Supports:
    - str identifiers
    - (name, params) tuples
    - pre-built estimator objects
    """

    estimators_params = estimators_params or {}

    def build_model(est_spec):
        if isinstance(est_spec, tuple):
            name, params = est_spec
            return EstimatorFactory.create(name, **(params or {}))
        if isinstance(est_spec, str):
            return EstimatorFactory.create(
                est_spec,
                **estimators_params.get(est_spec, {})
            )
        return est_spec
    
    def fit_one(est_spec):
        model = build_model(est_spec)
        model.fit(X, y)
        return model
    
    if n_jobs == 1:
        return [fit_one(est) for est in estimators]
    
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_one)(est) for est in estimators
    )

def predict_estimators(
    X: np.ndarray,
    estimators,
    n_jobs: int = 1,
):
    def predict_one(est):
        return est.predict(X)
    
    if n_jobs == 1:
        preds = [predict_one(est) for est in estimators]
        return np.column_stack(preds)
    
    preds = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(predict_one)(est) for est in estimators
    )
    return np.column_stack(preds)

def resolve_training_context(
    X: ArrayLike,
    y: ArrayLike,
    *,
    X_l: ArrayLike | None = None,
    y_l: ArrayLike | None = None,
    as_predictions: bool = False,
    splitter: BaseDataSplitter | None = None,
    split_ratio: float = 0.5,
    overlap: float = 0.0,
    random_state: int | None = None,
) -> TrainingContext:
    """
    Resolve COBRA training and aggregation datasets.

    This helper validates inputs and constructs the appropriate
    training/calibration context used throughout COBRA pipelines.

    Supported modes
    ---------------
    1. Prediction mode
        Inputs are already estimator predictions.

    2. Explicit split mode
        User provides both training and aggregation datasets.

    3. Automatic split mode
        Splitter generates train/calibration partitions automatically.

    Parameters
    ----------
    X : ArrayLike
        Input feature matrix or prediction matrix.

    y : ArrayLike
        Target values.

    X_l : ArrayLike | None, default=None
        Aggregation/calibration feature matrix.

    y_l : ArrayLike | None, default=None
        Aggregation/calibration targets.

    as_predictions : bool, default=False
        Whether ``X`` already contains estimator predictions.

    splitter : BaseDataSplitter | None, default=None
        Dataset splitter used for automatic partitioning.

        If ``None``, a default overlap splitter is created.

    Returns
    -------
    TrainingContext
        Resolved training/calibration context.

    Raises
    ------
    ValueError
        If only one of ``X_l`` or ``y_l`` is provided.

    Examples
    --------
    Automatic split:

    >>> ctx = resolve_training_context(X, y)

    Explicit split:

    >>> ctx = resolve_training_context(
    ...     X_k,
    ...     y_k,
    ...     X_l=X_l,
    ...     y_l=y_l,
    ... )

    Prediction mode:

    >>> ctx = resolve_training_context(
    ...     predictions,
    ...     y,
    ...     as_predictions=True,
    ... )
    """
    X, y = check_X_y(X, y)

    if as_predictions:
        return TrainingContext(
            X_k=None,
            y_k=None,
            X_l=np.asarray(X),
            y_l=np.asarray(y),
            as_predictions=True,
        )

    if (X_l is None) != (y_l is None):
        raise ValueError(
            "Both 'X_l' and 'y_l' must be provided together."
        )

    if X_l is not None and y_l is not None:
        X_l, y_l = check_X_y(X_l, y_l)

        return TrainingContext(
            X_k=np.asarray(X),
            y_k=np.asarray(y),
            X_l=np.asarray(X_l),
            y_l=np.asarray(y_l),
            as_predictions=False,
        )

    if splitter is None:
        splitter = SplitterFactory.create(
            "split_overlap",
            split_ratio=split_ratio,
            overlap=overlap,
            random_state=random_state,
        )

    split_indices: SplitIndices = splitter.split(X, y)

    train_idx = split_indices.train_idx
    eval_idx = split_indices.eval_idx

    return TrainingContext(
        X_k=np.asarray(X)[train_idx],
        y_k=np.asarray(y)[train_idx],
        X_l=np.asarray(X)[eval_idx],
        y_l=np.asarray(y)[eval_idx],
        as_predictions=False,
    )
    