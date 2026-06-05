"""
Cross-validation module for COBRA framework.

This module defines the abstract interface for cross-validation
strategies and provides a factory system for dynamic registration.

Cross-validation is used to:
- evaluate estimator generalization performance
- generate multiple train/evaluation splits
- support model selection in optimization loops
- stabilize COBRA aggregation across folds

In the COBRA pipeline, cross-validation sits above the splitter
layer and produces multiple SplitIndices objects for iterative training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory
from cobra.core.types import SplitIndices

class BaseCrossValidator(ABC):
    """
    Abstract base class for cross-validation strategies.

    Cross-validation generates multiple train/evaluation splits
    from a dataset to evaluate model performance across folds.

    This is a higher-level abstraction built on top of:
    - BaseDataSplitter (single split logic)

    Methods
    -------
    split(x, y, groups=None)
        Yield multiple SplitIndices objects.

    get_n_splits()
        Return number of folds/splits.
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
        Generate cross-validation splits.

        Parameters
        ----------
        x : ArrayLike
            Input features.

        y : ArrayLike
            Target values.

        groups : ArrayLike or None, optional
            Optional group labels for group-based CV.

        Yields
        ------
        SplitIndices
            Train/evaluation index splits for each fold.
        """
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self) -> int:
        """
        Return number of cross-validation folds.

        Returns
        -------
        int
            Number of splits.
        """
        raise NotImplementedError

class CVFactory(BaseFactory):
    """
    Factory for cross-validation strategies.

    Enables dynamic registration and instantiation of CV methods
    used in COBRA training and evaluation pipelines.
    """
    pass
