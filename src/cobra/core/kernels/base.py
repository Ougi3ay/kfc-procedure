"""
Kernel module for similarity weighting in the COBRA pipeline.

This module defines the kernel layer, which transforms adapted distance
matrices into similarity weights used for neighbor selection and
final aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
The kernel stage converts transformed distances into similarity scores
or weights that determine how much each neighbor contributes to the
final prediction.

Unlike distance metrics (which measure dissimilarity), kernels:

- convert distances into similarity space
- emphasize local neighborhoods
- control smoothness of the prediction function
- enable non-linear weighting of experts

Typical kernel outputs:

- similarity matrices
- weight distributions
- neighborhood importance scores

Design goals
------------
- modular kernel implementations
- interchangeable kernel functions
- compatibility with optimization routines
- factory-based instantiation for experiments

Examples
--------
>>> @KernelFactory.register("gaussian")
... class GaussianKernel(BaseKernel):
...     def __call__(self, distances):
...         return np.exp(-distances ** 2)

>>> kernel = KernelFactory.create("gaussian")
>>> weights = kernel(distance_matrix)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """
    Abstract base class for kernel functions.

    Kernels transform distance values into similarity weights used
    for neighbor influence modeling in the COBRA pipeline.

    Parameters
    ----------
    **kwargs : dict
        Kernel hyperparameters (e.g., bandwidth, gamma).

    Attributes
    ----------
    params : dict
        Stored kernel configuration parameters.

    Notes
    -----
    Subclasses must implement the ``__call__`` method.

    Kernels are typically used after the kernel adapter stage.

    Examples
    --------
    >>> class LinearKernel(BaseKernel):
    ...     def __call__(self, x):
    ...         return 1 - x
    """

    def __init__(self, **kwargs):
        """
        Initialize kernel with hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Kernel configuration parameters.
        """
        self.params = dict(kwargs)

        for key, value in self.params.items():
            setattr(self, key, value)

    def set_params(self, **params):
        """
        Update kernel parameters.

        Parameters
        ----------
        **params : dict
            Parameters to update.

        Returns
        -------
        BaseKernel
            Returns self for method chaining.

        Examples
        --------
        >>> kernel.set_params(gamma=0.1)
        """
        for key, value in params.items():
            setattr(self, key, value)
            self.params[key] = value

        return self

    def get_params(self, deep=True):
        """
        Return kernel parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn compatibility.

        Returns
        -------
        dict
            Kernel parameter dictionary.

        Examples
        --------
        >>> kernel.get_params()
        {'gamma': 0.5}
        """
        return dict(self.params)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Compute the kernel transformation on distance matrices.

        Parameters
        ----------
        distances : array-like
            One or more distance matrices. Typical shape is
            ``(n_queries, n_references)`` for a pairwise distance from a
            set of query samples to reference samples. Some kernels accept a
            single square distance matrix of shape ``(n_samples, n_samples)``.

        Returns
        -------
        np.ndarray
            Kernel-weighted similarity matrix. Output shape matches the
            primary distance input, typically ``(n_queries, n_references)``.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> weights = kernel(distances)
        """
        raise NotImplementedError


class KernelFactory(BaseFactory):
    """
    Factory for kernel implementations.

    This registry-based factory enables dynamic creation of kernel
    functions using string identifiers.

    It is used in:

    - COBRA-style ensemble pipelines
    - hyperparameter optimization
    - YAML-based configuration systems
    - kernel benchmarking experiments

    Examples
    --------
    >>> kernel = KernelFactory.create("gaussian")

    >>> weights = kernel(distance_matrix)
    """
    pass