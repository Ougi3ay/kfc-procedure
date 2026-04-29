"""
Kernel adapter module for transforming distance outputs before kernel evaluation.

This module defines the ``BaseKernelAdapter`` abstraction and the
``KernelAdapterFactory`` registry used to manage adapter implementations.

Kernel adapters act as an intermediate layer between the distance computation
stage and the kernel function stage in the model pipeline.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
The kernel adapter is responsible for injecting tunable hyperparameters
into raw distance values before they are passed to the kernel function.

This design is especially useful during optimization, where parameters such as:

- alpha (prediction-space weight)
- beta (input-space weight)
- bandwidth

must be adjusted without modifying the distance computation logic itself.

By separating this logic into adapters, the architecture becomes:

- modular
- extensible
- optimization-friendly
- easier to test and maintain

Examples
--------
>>> @KernelAdapterFactory.register("mixcobra")
... class MixCOBRAKernelAdapter(BaseKernelAdapter):
...     def __init__(self, alpha=1.0, beta=0.0):
...         super().__init__(alpha=alpha, beta=beta)
...
...     def transform(self, pred_distance, input_distance):
...         return self.alpha * pred_distance + self.beta * input_distance

>>> adapter = KernelAdapterFactory.create(
...     "mixcobra",
...     alpha=2.0,
...     beta=0.5
... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernelAdapter(ABC):
    """
    Abstract base class for kernel adapters.

    A kernel adapter transforms one or more distance matrices into a
    kernel-ready representation by applying tunable hyperparameters.

    This abstraction separates:

    - distance computation
    - hyperparameter injection
    - kernel evaluation

    allowing optimization routines to tune parameters independently.

    Parameters
    ----------
    **kwargs : dict
        Hyperparameters used by the adapter.

    Attributes
    ----------
    params : dict
        Internal dictionary storing adapter parameters.

    Notes
    -----
    Subclasses must implement the ``transform()`` method.

    The ``transform()`` method may accept one or multiple distance arrays,
    depending on the aggregation strategy.

    Examples
    --------
    >>> class SimpleAdapter(BaseKernelAdapter):
    ...     def transform(self, distance):
    ...         return self.alpha * distance
    """

    def __init__(self, **kwargs):
        """
        Initialize adapter with hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Adapter configuration parameters.
        """
        self.params = dict(kwargs)
        self.set_params(**kwargs)

    def set_params(self, **params):
        """
        Update adapter parameters.

        Parameters
        ----------
        **params : dict
            Parameters to update.

        Returns
        -------
        BaseKernelAdapter
            Returns self for method chaining.

        Examples
        --------
        >>> adapter.set_params(alpha=2.0)
        """
        for k, v in params.items():
            setattr(self, k, v)

        self.params.update(params)
        return self

    def get_params(self, deep=True):
        """
        Return adapter parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn compatibility.
            Currently not used.

        Returns
        -------
        dict
            Dictionary of stored parameters.

        Examples
        --------
        >>> adapter.get_params()
        {'alpha': 1.0}
        """
        return dict(self.params)

    @abstractmethod
    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Transform distance matrices before kernel evaluation.

        The adapter layer combines or rescales one or more distance
        matrices into a kernel-ready representation. Common usages include
        linear mixing (alpha*pred + beta*input) or applying a learned
        transformation to a single distance matrix.

        Parameters
        ----------
        *distances : np.ndarray
            One or more distance matrices. Each matrix is expected to have
            shape ``(n_queries, n_references)`` or ``(n_samples, n_samples)``
            for square pairwise distances.

        Returns
        -------
        np.ndarray
            Transformed distance matrix ready for kernel computation. Shape
            should align with the primary distance input (typically
            ``(n_queries, n_references)``).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> adapter.transform(pred_distance, input_distance)
        """
        raise NotImplementedError


class KernelAdapterFactory(BaseFactory):
    """
    Factory for ``BaseKernelAdapter`` implementations.

    This factory enables dynamic registration and instantiation of
    kernel adapter classes using string identifiers.

    It is commonly used inside configurable pipelines and YAML-based
    model construction systems.

    Examples
    --------
    >>> adapter = KernelAdapterFactory.create(
    ...     "mixcobra",
    ...     alpha=1.0,
    ...     beta=0.5
    ... )
    """
    pass