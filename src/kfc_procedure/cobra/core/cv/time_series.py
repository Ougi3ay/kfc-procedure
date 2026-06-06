"""
Time Series Cross Validation.

This method preserves temporal ordering of data and prevents
future information leakage.

Training sets grow over time while validation sets move forward
chronologically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseCrossValidator, CVFactory
from kfc_procedure.cobra.core.types import SplitIndices


@CVFactory.register("time_series", "tscv")
class TimeSeriesCV(BaseCrossValidator):
    """
    Time Series Cross Validation.

    Parameters
    ----------
    n_splits : int
        Number of splits.

    test_size : int or None
        Size of each validation window.
    """

    def __init__(self, n_splits: int = 5, test_size: int | None = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, x: ArrayLike, y: ArrayLike):
        n = len(x)

        step = self.test_size or (n // (self.n_splits + 1))

        for i in range(self.n_splits):
            train_end = step * (i + 1)
            val_end = step * (i + 2)

            train_idx = np.arange(train_end)
            val_idx = np.arange(train_end, min(val_end, n))

            yield SplitIndices(
                train_idx=train_idx,
                eval_idx=val_idx,
                fold_id=i,
            )

    def get_n_splits(self) -> int:
        return self.n_splits
