"""
Data splitting module for COBRA training and calibration workflows.

This module defines the splitting layer used to partition datasets
into training and calibration (or validation) subsets while preserving
index alignment.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Data splitting is a foundational step in COBRA pipelines, ensuring that
estimators and calibration procedures operate on disjoint subsets of data.

Typical roles include:

- separating training data for estimator fitting
- reserving calibration data for consensus weighting
- maintaining index consistency across all pipeline stages
- enabling reproducible experimental setups

Unlike standard ML splits, COBRA-style splitting emphasizes:

- index-based partitioning (not just data copies)
- alignment across multiple estimator outputs
- compatibility with ensemble calibration logic

Design goals
------------
- provide consistent split interface
- support multiple splitting strategies
- ensure reproducibility of index assignments
- integrate with factory-based pipeline configuration
- avoid data mutation (index-based design preferred)

Examples
--------
>>> @SplitterFactory.register("holdout")
... class HoldoutSplitter(BaseDataSplitter):
...     def split(self, x, y):
...         n = len(x)
...         idx = np.random.permutation(n)
...         return idx[:int(0.8*n)], idx[int(0.8*n):]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseDataSplitter(ABC):
    """
    Abstract base class for dataset splitting strategies.

    This interface defines how datasets are partitioned into subsets
    used for training and calibration in COBRA pipelines.

    Pipeline role
    -------------
    Splitters determine:

    - which samples are used for estimator training
    - which samples are used for calibration
    - how indices are preserved across pipeline stages

    Notes
    -----
    Implementations must return index arrays rather than raw data
    to ensure consistency across estimator pools.

    Examples
    --------
    >>> class RandomSplitter(BaseDataSplitter):
    ...     def split(self, x, y):
    ...         return np.array([0,1]), np.array([2,3])
    """

    @abstractmethod
    def split(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split dataset into training and calibration indices.

        Parameters
        ----------
        x : ArrayLike
            Input feature matrix.

        y : ArrayLike
            Target values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - train_indices
            - calibration_indices

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> train_idx, calib_idx = splitter.split(X, y)
        """
        raise NotImplementedError


class SplitterFactory(BaseFactory):
    """
    Factory for dataset splitter implementations.

    This registry-based factory enables dynamic selection of dataset
    partitioning strategies used in COBRA pipelines.

    It is commonly used in:

    - train/calibration splitting
    - ensemble validation setups
    - reproducible experiment design
    - configuration-driven pipelines

    Examples
    --------
    >>> splitter = SplitterFactory.create("holdout")

    >>> train_idx, calib_idx = splitter.split(X, y)
    """
    pass