


from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(slots=True)
class SplitIndices:
    """
    Container for dataset index partitions.

    Attributes
    ----------
    train_idx : np.ndarray
        Training subset indices.
    eval_idx : np.ndarray
        Evaluation subset indices.
    """

    train_idx: np.ndarray
    eval_idx: np.ndarray
    fold_id: int | None = None

@dataclass(slots=True)
class TrainingContext:
    """
    Container for resolved training/calibration data.
    """
    X_k: np.ndarray | None
    y_k: np.ndarray | None

    X_l: np.ndarray
    y_l: np.ndarray

    as_predictions: bool
