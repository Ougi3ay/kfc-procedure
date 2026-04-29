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

from cobra.core.aggregators.base import AggregatorFactory
from cobra.core.distances.base import DistanceFactory
from cobra.core.estimators.base import EstimatorFactory
from cobra.core.kernels.base import KernelFactory
from cobra.core.losses.base import LossFactory
from cobra.core.splitters.base import SplitterFactory


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
