"""
Concrete dataset splitters for COBRA-style training and calibration workflows.

This module implements common dataset partitioning strategies used in
COBRA pipelines, including holdout, K-fold cross-validation, and
overlapping splits.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Splitters define how input data is partitioned into subsets used for:

- estimator training
- calibration / aggregation
- cross-validation
- ensemble evaluation

In COBRA-style systems, splitting is index-based to ensure:

- consistent alignment across estimators
- reproducible experimental design
- compatibility with ensemble aggregation logic

Available strategies
--------------------

1. RandomHoldoutSplitter
^^^^^^^^^^^^^^^^^^^^^^^^

Randomly partitions data into training and calibration sets.

Typical use cases:
- simple train/calibration split
- baseline COBRA experiments

2. KFoldSplitter
^^^^^^^^^^^^^^^^

Generates K-fold cross-validation splits.

Typical use cases:
- model evaluation
- robust performance estimation
- ensemble validation

3. OverlapSplitter
^^^^^^^^^^^^^^^^^^

Creates overlapping training and calibration sets.

Typical use cases:
- soft calibration setups
- robustness experiments
- MIXCOBRA-style overlap learning

Design goals
------------
- index-preserving splitting (no data duplication)
- reproducible randomness via seeds
- flexible split strategies
- compatibility with ensemble pipelines
- support for advanced overlap calibration

Examples
--------
>>> splitter = SplitterFactory.create("holdout")
>>> train_idx, cal_idx = splitter.split(X, y)

>>> splitter = SplitterFactory.create("kfold")
>>> folds = splitter.split(X, y)

>>> splitter = SplitterFactory.create("split_overlap")
>>> train_idx, agg_idx = splitter.split(X, y)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split

from cobra.utils.preprocessing import data_split_overlap

from .base import BaseDataSplitter, SplitterFactory


@SplitterFactory.register("holdout", "random_holdout")
class RandomHoldoutSplitter(BaseDataSplitter):
    """
    Random holdout splitter.

    Splits dataset into training and calibration subsets using a
    randomized partition.

    Parameters
    ----------
    calibration_size : float, default=0.5
        Fraction of samples used for calibration.

    random_state : int | None
        Seed for reproducibility.

    Notes
    -----
    This is the simplest COBRA-compatible split strategy.

    Examples
    --------
    >>> splitter = RandomHoldoutSplitter(0.3)
    >>> train_idx, cal_idx = splitter.split(X, y)
    """

    def __init__(self, calibration_size: float = 0.5, random_state: int | None = None) -> None:
        self.calibration_size = float(calibration_size)
        self.random_state = random_state

    def split(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Split dataset into train and calibration indices.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            train_idx, calibration_idx
        """
        n_samples = np.asarray(x).shape[0]
        indices = np.arange(n_samples)

        train_idx, cal_idx = train_test_split(
            indices,
            test_size=self.calibration_size,
            random_state=self.random_state,
            shuffle=True,
        )

        return np.asarray(train_idx), np.asarray(cal_idx)


@SplitterFactory.register("kfold")
class KFoldSplitter(BaseDataSplitter):
    """
    K-fold cross-validation splitter.

    Produces multiple train/validation splits for robust evaluation.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.

    shuffle : bool, default=True
        Whether to shuffle data before splitting.

    random_state : int | None
        Seed for reproducibility.

    Notes
    -----
    Unlike holdout, this returns multiple splits.

    Examples
    --------
    >>> splitter = KFoldSplitter(n_splits=5)
    >>> folds = splitter.split(X, y)
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x: ArrayLike, y: ArrayLike):
        """
        Generate K-fold indices.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_idx, val_idx)
        """
        n_samples = np.asarray(x).shape[0]
        indices = np.arange(n_samples)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        return [
            (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64),
            )
            for train_idx, val_idx in kf.split(indices)
        ]


@SplitterFactory.register("split_overlap")
class OverlapSplitter(BaseDataSplitter):
    """
    Overlapping dataset splitter.

    Creates partially overlapping training and calibration sets,
    useful for soft-consensus and MIXCOBRA-style learning.

    Parameters
    ----------
    split_ratio : float, default=0.5
        Base fraction of dataset assigned to training.

    overlap : float, default=0.0
        Fraction of overlap between train and calibration sets.

    shuffle : bool, default=True
        Whether to shuffle indices before splitting.

    random_state : int | None
        Seed for reproducibility.

    Notes
    -----
    - Higher overlap increases shared samples between sets
    - Useful for smoother consensus estimation

    Examples
    --------
    >>> splitter = OverlapSplitter(split_ratio=0.6, overlap=0.1)
    >>> train_idx, cal_idx = splitter.split(X, y)
    """

    def __init__(
        self,
        split_ratio: float = 0.5,
        overlap: float = 0.0,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.split_ratio = float(split_ratio)
        self.overlap = float(overlap)
        self.shuffle = shuffle
        self.random_state = random_state

    def _shuffle_indices(self, indices: np.ndarray) -> np.ndarray:
        """Shuffle indices deterministically."""
        if not self.shuffle:
            return indices

        rng = np.random.default_rng(self.random_state)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled

    def split(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate overlapping train and calibration indices.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            train_idx, calibration_idx
        """
        n = np.asarray(x).shape[0]
        indices = np.arange(n)

        indices = self._shuffle_indices(indices)

        if not (0 < self.split_ratio < 1):
            raise ValueError(f"`split_ratio` must be in (0,1), got {self.split_ratio}")

        if not (0 <= self.overlap < 1):
            raise ValueError(f"`overlap` must be in [0,1), got {self.overlap}")

        if self.overlap >= self.split_ratio:
            raise ValueError("`overlap` must be smaller than `split_ratio`")

        k1 = int(n * (self.split_ratio - self.overlap / 2))
        k2 = int(n * (self.split_ratio + self.overlap / 2))

        k1 = max(0, min(k1, n))
        k2 = max(0, min(k2, n))

        idx_train = indices[:k2].astype(np.int64)
        idx_agg = indices[k1:].astype(np.int64)

        return idx_train, idx_agg