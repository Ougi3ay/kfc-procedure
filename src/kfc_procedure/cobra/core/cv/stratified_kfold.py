"""
Stratified K-Fold Cross Validation.

This strategy preserves class distribution across all folds.
It is especially important for imbalanced classification problems.

Each fold maintains approximately the same label proportions
as the original dataset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseCrossValidator, CVFactory
from kfc_procedure.cobra.core.types import SplitIndices


@CVFactory.register("stratified_kfold")
class StratifiedKFoldCV(BaseCrossValidator):
    """
    Stratified K-Fold Cross Validation.

    Parameters
    ----------
    n_splits : int
        Number of folds.

    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_splits: int = 5, random_state: int | None = None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)

        class_map = {}
        for idx, label in enumerate(y):
            class_map.setdefault(label, []).append(idx)

        folds = [[] for _ in range(self.n_splits)]

        for label, idxs in class_map.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            parts = np.array_split(idxs, self.n_splits)

            for i in range(self.n_splits):
                folds[i].extend(parts[i])

        folds = [np.array(f) for f in folds]

        for i in range(self.n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])

            yield SplitIndices(
                train_idx=train_idx,
                eval_idx=val_idx,
                fold_id=i,
            )

    def get_n_splits(self) -> int:
        return self.n_splits
