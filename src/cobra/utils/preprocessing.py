"""
Data preprocessing utilities for COBRA pipeline construction.

This module provides helper functions for dataset splitting and
normalization constant computation used across COBRA-style systems.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Preprocessing utilities ensure that raw datasets are properly prepared
for downstream COBRA components by handling:

- overlapping dataset splits
- index alignment between sub-datasets
- normalization constant computation
- scale stabilization for distance and kernel stages

These utilities are essential for maintaining numerical stability
and consistent behavior across heterogeneous estimators.

Design goals
------------
- provide reusable preprocessing primitives
- support COBRA and MIXCOBRA overlap strategies
- ensure deterministic and reproducible splits
- stabilize scaling across model outputs
- remain lightweight and dependency-aware (NumPy / sklearn only)

Functions
---------
- ``data_split_overlap``
    Creates overlapping dataset partitions for calibration workflows.

- ``compute_normalization_constant``
    Computes scaling constants for stabilizing feature magnitudes.

Examples
--------
>>> X_k, y_k, X_l, y_l, idx_k, idx_l = data_split_overlap(X, y)

>>> c = compute_normalization_constant(X)
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle


def data_split_overlap(
    X: np.ndarray,
    y: np.ndarray,
    split: float = 0.5,
    overlap: float = 0.0,
    shuffle: bool = True,
    random_state: int = None,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Split dataset into overlapping partitions.

    This function divides data into two subsets (D_k and D_l) with a
    controllable overlap, useful for COBRA-style calibration and
    ensemble learning.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    y : np.ndarray
        Target vector.

    split : float, default=0.5
        Base proportion of data assigned to the first split.

    overlap : float, default=0.0
        Fraction of overlap between the two splits.

    shuffle : bool, default=True
        Whether to shuffle indices before splitting.

    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_k, y_k, X_l, y_l, iloc_k, iloc_l)

        - X_k, y_k : first subset
        - X_l, y_l : second subset
        - iloc_k, iloc_l : index arrays

    Raises
    ------
    ValueError
        If split or overlap parameters are invalid.

    Notes
    -----
    - Increasing overlap increases shared samples between splits
    - Used in MIXCOBRA-style calibration strategies
    """
    if not (0 < split < 1):
        raise ValueError(f"`split` must be in (0,1), got {split}")

    if not (0 <= overlap < 1):
        raise ValueError(f"`overlap` must be in [0,1), got {overlap}")

    if overlap >= split:
        raise ValueError("`overlap` must be smaller than `split`")

    n = len(y)
    indices = np.arange(n)

    if shuffle:
        indices = sklearn_shuffle(indices, random_state=random_state)

    k1 = int(n * (split - overlap / 2))
    k2 = int(n * (split + overlap / 2))

    k1 = max(0, min(k1, n))
    k2 = max(0, min(k2, n))

    iloc_k = indices[:k2].astype(np.int64)
    iloc_l = indices[k1:].astype(np.int64)

    X_k, y_k = X[iloc_k], y[iloc_k]
    X_l, y_l = X[iloc_l], y[iloc_l]

    return X_k, y_k, X_l, y_l, iloc_k, iloc_l


def compute_normalization_constant(
    y: np.ndarray,
    norm_constant: float | None = None,
    scale_factor: float = 30.0,
    M: int = 1,
) -> float:
    """
    Compute normalization constant.

    The normalization constant is used to rescale prediction outputs
    before distance computation in the aggregation space.

    Formula
    -------
    c = scale_factor / (max(abs(y)) * M)

    where:

    - ``y`` is the target vector,
    - ``M`` is the number of estimators,
    - ``scale_factor`` controls the global scaling magnitude.

    Parameters
    ----------
    y : np.ndarray
        Target values used to compute the scaling factor.

    norm_constant : float | None, default=None
        Optional predefined normalization numerator.
        If provided, replaces ``scale_factor`` in the formula.

    scale_factor : float, default=30.0
        Default scaling numerator used when
        ``norm_constant`` is not provided.

    M : int, default=1
        Number of estimators used in the prediction space.

    Returns
    -------
    float
        Normalization constant applied to prediction outputs.

    Notes
    -----
    The scaling is inversely proportional to the maximum absolute
    target magnitude and the number of estimators, helping stabilize
    distance computations in the aggregation space.

    This follows the original GradientCOBRA normalization strategy:

    ``predictions_scaled = predictions * c``
    """
    max_val = np.max(np.abs(y)) + 1e-12

    c = (
        norm_constant
        if norm_constant is not None
        else scale_factor
    )

    return c / (max_val * M)

def clean_sklearn_name(name: str) -> str:
    """
    Convert sklearn class name to snake_case factory key.

    Examples:
        LogisticRegression -> logistic_regression
        RandomForestRegressor -> random_forest_regressor
        SVC -> svc
    """
    # Step 1: handle acronyms (SVC, SVR, PCA)
    if name.isupper():
        return name.lower()

    # Step 2: convert CamelCase → snake_case
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)

    return s2.lower()