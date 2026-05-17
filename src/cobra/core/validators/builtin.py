"""
K-Fold cross-validation splitter."""

from __future__ import annotations
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.types import SplitIndices
from cobra.core.validators.base import BaseCrossValidator, CVFactory


@CVFactory.register("kfold")
class KFoldCV(BaseCrossValidator):
    """
    K-Fold cross-validation splitter.

    Generates K disjoint validation folds while using the remaining
    samples for training.

    This is a standard evaluation strategy in COBRA pipelines for:
    - model evaluation
    - estimator comparison
    - robust performance estimation
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

    def get_n_splits(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> int:
        return self.n_splits

    def split(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Iterator[SplitIndices]:

        n = len(x)

        rng = np.random.RandomState(self.random_state)

        indices = np.arange(n)

        if self.shuffle:
            indices = rng.permutation(indices)
        
        folds = [[] for _ in range(self.n_splits)]

        for i, idx in enumerate(indices):
            folds[i % self.n_splits].append(idx)

        folds = [np.array(f, dtype=np.int64) for f in folds]
        
        for i in range(self.n_splits):
            val_idx = folds[i]

            train_idx = np.concatenate(
                [folds[j] for j in range(self.n_splits) if j != i]
            )

            train_idx = np.sort(train_idx)

            yield SplitIndices(
                train_idx=train_idx,
                eval_idx=val_idx,
                fold_id=i,
            )

