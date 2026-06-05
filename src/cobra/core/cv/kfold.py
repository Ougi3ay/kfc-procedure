"""
K-Fold cross-validation splitter.

This module implements K-Fold cross-validation, a standard evaluation
strategy that splits data into K disjoint folds.

Each fold is used once as a validation set, while the remaining
folds are used for training.

This strategy is widely used in COBRA pipelines for:
- estimator evaluation
- model selection
- robust performance estimation
- reducing variance in evaluation metrics
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.types import SplitIndices
from cobra.core.cv.base import BaseCrossValidator, CVFactory


@CVFactory.register("kfold")
class KFoldCV(BaseCrossValidator):
    """
    K-Fold Cross-Validation.

    This class implements standard K-Fold splitting where the dataset
    is randomly partitioned into K folds.

    Parameters
    ----------
    n_splits : int
        Number of folds (K).

    shuffle : bool
        Whether to shuffle indices before splitting.

    random_state : int or None
        Random seed for reproducibility.

    Notes
    -----
    - Each sample appears exactly once in a validation fold.
    - Training folds are complementary to validation folds.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Iterator[SplitIndices]:

        n = len(x)
        indices = np.arange(n)

        rng = np.random.RandomState(self.random_state)

        if self.shuffle:
            indices = rng.permutation(indices)

        # distribute indices into folds
        folds = [[] for _ in range(self.n_splits)]

        for i, idx in enumerate(indices):
            folds[i % self.n_splits].append(idx)

        folds = [np.asarray(f, dtype=np.int64) for f in folds]

        # generate train/validation splits
        for i in range(self.n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])

            train_idx = np.sort(train_idx)

            yield SplitIndices(
                train_idx=train_idx,
                eval_idx=val_idx,
                fold_id=i,
            )
