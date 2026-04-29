"""
Search optimization module for discrete and stochastic hyperparameter tuning.

This module defines the search-based optimization layer used in the
COBRA pipeline for exploring discrete or structured parameter spaces.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Search optimizers are used when parameters are:

- discrete (e.g., number of neighbors, tree depth)
- categorical (e.g., kernel type, estimator type)
- structured (e.g., model configurations)
- non-differentiable or black-box

Unlike gradient-based methods, search optimizers rely on:

- sampling strategies
- grid exploration
- random search
- heuristic exploration

Design goals
------------
- unify discrete search strategies under one interface
- support flexible search space definitions
- enable reproducible sampling
- integrate with COBRA objective evaluation
- support factory-based configuration

Examples
--------
>>> @SearchOptimizerFactory.register("random_search")
... class RandomSearch(BaseSearchOptimizer):
...     def search_space(self):
...         return {"lr": [0.01, 0.1], "gamma": [0.5, 1.0]}
...
...     def sample(self):
...         return {"lr": np.random.choice(self.search_space()["lr"])}
"""

from __future__ import annotations

from abc import abstractmethod

from cobra.core.factory import BaseFactory
from cobra.core.optimizers.base import BaseOptimizer


class BaseSearchOptimizer(BaseOptimizer):
    """
    Abstract base class for search-based optimizers.

    This class defines the interface for optimizers that explore
    discrete or categorical parameter spaces rather than using
    gradients.

    Pipeline role
    -------------
    Search optimizers tune:

    - estimator configurations
    - kernel types
    - distance metrics
    - discrete hyperparameters

    Notes
    -----
    Subclasses must implement:

    - ``search_space()``: defines available parameter values
    - ``sample()``: draws a candidate configuration

    These methods are typically used in random search, grid search,
    or evolutionary strategies.

    Examples
    --------
    >>> class GridSearch(BaseSearchOptimizer):
    ...     def search_space(self):
    ...         return {"k": [3, 5, 7]}
    ...
    ...     def sample(self):
    ...         return {"k": 3}
    """

    @abstractmethod
    def search_space(self):
        """
        Define the search space of hyperparameters.

        Returns
        -------
        dict
            Mapping from parameter names to possible values.

        Examples
        --------
        >>> {"gamma": [0.1, 0.5, 1.0]}
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """
        Sample a single configuration from the search space.

        Returns
        -------
        dict
            A sampled hyperparameter configuration.

        Examples
        --------
        >>> {"gamma": 0.5, "lr": 0.01}
        """
        raise NotImplementedError


class SearchOptimizerFactory(BaseFactory):
    """
    Factory for search-based optimizer implementations.

    This registry-based factory enables dynamic selection of discrete
    search strategies for hyperparameter optimization.

    It is commonly used in:

    - grid search pipelines
    - random search experiments
    - architecture selection
    - COBRA configuration tuning

    Examples
    --------
    >>> optimizer = SearchOptimizerFactory.create("random_search")

    >>> config = optimizer.sample()
    """
    pass
