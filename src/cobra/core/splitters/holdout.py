"""
Random holdout data splitting strategy.

This module implements a random holdout partitioning scheme used to
separate a dataset into training and calibration subsets.

The resulting partitions are commonly employed by COBRA-based
aggregation methods, where base estimators are trained on one subset
and aggregation is performed using predictions generated on an
independent calibration subset.

The implementation relies on random sampling without replacement,
ensuring that observations belong exclusively to either the training
or calibration partition.
"""

from __future__ import annotations
import numpy as np
from .base import BaseDataSplitter, SplitterFactory

from cobra.core.types import SplitIndices
from sklearn.model_selection import train_test_split

@SplitterFactory.register("holdout", "random_holdout")
class RandomHoldoutSplitter(BaseDataSplitter):
    """
    Random holdout data splitter.

    This splitter randomly partitions a dataset into two disjoint
    subsets:

    * Training subset
    * Calibration subset

    The training subset is typically used to fit base estimators,
    while the calibration subset is reserved for aggregation,
    validation, or prediction-combination procedures.

    Parameters
    ----------
    calibration_size : float, default=0.5
        Fraction of observations assigned to the calibration
        subset. Must lie in the interval ``(0, 1)``.

    random_state : int or None, default=None
        Seed controlling the randomness of the partition.
        Providing a value ensures reproducible splits.

    Notes
    -----
    The split is generated using
    :func:`sklearn.model_selection.train_test_split`.

    No sample can simultaneously belong to both subsets.
    """

    def __init__(
        self,
        calibration_size: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        self.calibration_size = float(calibration_size)
        self.random_state = random_state

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> SplitIndices:
        """
        Generate a random train/calibration partition.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        y : ndarray of shape (n_samples,)
            Target values associated with the observations.

        Returns
        -------
        SplitIndices
            Object containing the indices assigned to the
            training and calibration subsets.

        Raises
        ------
        ValueError
            If ``calibration_size`` is invalid or the input data
            cannot be partitioned.

        Notes
        -----
        Only the number of samples is used to generate the split.
        The feature matrix and target vector are not modified.
        """
        n_samples = np.asarray(x).shape[0]
        indices = np.arange(n_samples)
        train_idx, cal_idx = train_test_split(
            indices,
            test_size=self.calibration_size,
            random_state=self.random_state,
            shuffle=True,
        )

        return SplitIndices(
            train_idx=np.asarray(train_idx),
            eval_idx=np.asarray(cal_idx),
        )
