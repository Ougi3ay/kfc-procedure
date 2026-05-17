"""
Cross-validation module for COBRA evaluation and model selection workflows.

This module defines the cross-validation layer used to generate
multiple train/validation index partitions for robust estimator
evaluation and hyperparameter selection.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> CrossValidator -> Metrics
-> Aggregation -> Output

Purpose
-------
Cross-validation enables repeated evaluation of COBRA estimators
across multiple dataset partitions to improve robustness and reduce
evaluation variance.

Typical roles include:

- model evaluation
- hyperparameter optimization
- estimator comparison
- robustness analysis
- reproducible benchmarking

Unlike simple train/calibration splitting, cross-validation produces
multiple index partitions (folds) that systematically rotate validation
subsets across the dataset.

Design goals
------------
- provide consistent cross-validation interface
- support multiple CV strategies
- preserve index alignment across estimators
- ensure reproducibility
- integrate with factory-driven pipeline configuration
- avoid data mutation (index-based design preferred)

Examples
--------
>>> @CVFactory.register("kfold")
... class KFoldValidator(BaseCrossValidator):
...     def split(self, x, y):
...         n = len(x)
...         idx = np.arange(n)
...         fold_size = n // 5
...
...         for i in range(5):
...             start = i * fold_size
...             end = start + fold_size
...
...             val_idx = idx[start:end]
...             train_idx = np.concatenate([idx[:start], idx[end:]])
...
...             yield train_idx, val_idx
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory
from cobra.core.types import SplitIndices


class BaseCrossValidator(ABC):
    """
    Abstract base class for cross-validation strategies.

    This interface defines how datasets are partitioned into
    multiple train/validation folds for evaluation workflows
    in COBRA pipelines.

    Pipeline role
    -------------
    Cross-validators determine:

    - how validation folds are generated
    - how training subsets rotate across folds
    - how index consistency is preserved
    - how evaluation procedures are repeated

    Notes
    -----
    Implementations must yield index arrays rather than raw data
    to maintain alignment across estimator pools and calibration
    components.

    Unlike standard data splitters, cross-validators generate
    multiple partitions through iteration.

    Examples
    --------
    >>> class DummyCV(BaseCrossValidator):
    ...     def split(self, x, y):
    ...         yield np.array([0, 1]), np.array([2, 3])
    """

    @abstractmethod
    def split(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
    ) -> Iterator[SplitIndices]:
        """
        Generate train/validation index folds.

        Parameters
        ----------
        x : ArrayLike
            Input feature matrix.

        y : ArrayLike
            Target values.

        groups : ArrayLike | None
            Optional group labels for stratified splitting.

        Yields
        ------
        Iterator[SplitIndices]
            Iterator producing:

            - train_idx
            - eval_idx

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> for split_indices in cv.split(X, y):
        ...     train_idx = split_indices.train_idx
        ...     eval_idx = split_indices.eval_idx
        """
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self) -> int:
        """
        Return the number of generated folds.
        """
        raise NotImplementedError


class CVFactory(BaseFactory):
    """
    Factory for cross-validation implementations.

    This registry-based factory enables dynamic selection of
    cross-validation strategies used in COBRA evaluation pipelines.

    It is commonly used in:

    - model evaluation
    - hyperparameter tuning
    - estimator benchmarking
    - reproducible experimentation
    - configuration-driven workflows

    Examples
    --------
    >>> cv = CVFactory.create("kfold")

    >>> for split_indices in cv.split(X, y):
    ...     train_idx = split_indices.train_idx
    ...     eval_idx = split_indices.eval_idx
    """

    pass