"""
Overlapping train/evaluation splitting strategy.

This module implements a partitioning scheme in which the training
and evaluation subsets may share a controlled fraction of samples.

Unlike conventional holdout approaches that produce disjoint
partitions, the overlap splitter allows observations to appear in
both subsets. The amount of shared data is controlled through the
``overlap`` parameter.

This strategy is particularly useful for aggregation-based learning
frameworks where a balance between estimator training and evaluation
coverage is desired.
"""

from __future__ import annotations

import numpy as np

from cobra.core.types import SplitIndices
from .base import BaseDataSplitter, SplitterFactory


@SplitterFactory.register("split_overlap")
class OverlapSplitter(BaseDataSplitter):
    """
    Splitter supporting overlapping partitions.

    This splitter divides a dataset into training and evaluation
    subsets while allowing a configurable proportion of samples
    to belong to both partitions.

    Parameters
    ----------
    split_ratio : float, default=0.5
        Proportion of samples assigned to the primary training
        partition.

    overlap : float, default=0.0
        Fraction of observations shared between the training and
        evaluation subsets.

        - ``0.0`` produces non-overlapping partitions.
        - Larger values increase the number of shared samples.

    shuffle : bool, default=True
        Whether sample indices should be randomized before
        partitioning.

    random_state : int or None, default=None
        Seed controlling reproducibility when shuffling is enabled.

    Notes
    -----
    Let ``n`` denote the number of observations.

    The partition boundaries are computed as

    k₁ = n · (split_ratio − overlap / 2)

    k₂ = n · (split_ratio + overlap / 2)

    The resulting subsets are then defined as:

    - Training indices: ``[0, k₂)``
    - Evaluation indices: ``[k₁, n)``

    Samples lying between ``k₁`` and ``k₂`` belong to both
    partitions and constitute the overlap region.

    The overlap fraction must satisfy:

    - ``0 ≤ overlap < 1``
    - ``overlap < split_ratio``
    """

    def __init__(
        self,
        split_ratio: float = 0.5,
        overlap: float = 0.0,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        
        if not (0 < split_ratio < 1):
            raise ValueError("split_ratio must be in (0,1)")

        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0,1)")

        if overlap >= split_ratio:
            print(f"Invalid parameters: split_ratio={split_ratio}, overlap={overlap}")
            raise ValueError("overlap must be smaller than split_ratio")
        
        self.split_ratio = float(split_ratio)
        self.overlap = float(overlap)
        self.shuffle = shuffle
        self.random_state = random_state

    def _shuffle_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Randomly permute sample indices.

        Parameters
        ----------
        indices : ndarray of shape (n_samples,)
            Input index sequence.

        Returns
        -------
        ndarray
            Shuffled indices if ``shuffle=True``; otherwise the
            original indices.

        Notes
        -----
        Shuffling is performed using NumPy's random number generator
        initialized with ``random_state``.
        """
        if not self.shuffle:
            return indices

        rng = np.random.default_rng(self.random_state)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled

    def split(self, x: np.ndarray, y: np.ndarray) -> SplitIndices:
        """
        Generate overlapping training and evaluation partitions.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        y : ndarray of shape (n_samples,)
            Target values associated with the observations.

        Returns
        -------
        SplitIndices
            Object containing the training and evaluation indices.

        Raises
        ------
        ValueError
            If ``split_ratio`` is not in ``(0, 1)``.

        ValueError
            If ``overlap`` is not in ``[0, 1)``.

        ValueError
            If ``overlap`` is greater than or equal to
            ``split_ratio``.

        Notes
        -----
        The partition is constructed solely from sample indices.
        Neither the feature matrix nor the target vector is modified.

        Depending on the overlap fraction, some observations may
        belong to both the training and evaluation subsets.
        """
        n = np.asarray(x).shape[0]
        indices = np.arange(n)
        indices = self._shuffle_indices(indices)

        k1 = int(n * (self.split_ratio - self.overlap / 2))
        k2 = int(n * (self.split_ratio + self.overlap / 2))

        k1 = max(0, min(k1, n))
        k2 = max(0, min(k2, n))

        train_idx = indices[:k2]
        eval_idx = indices[k1:]

        return SplitIndices(
            train_idx=train_idx.astype(np.int64),
            eval_idx=eval_idx.astype(np.int64),
        )
