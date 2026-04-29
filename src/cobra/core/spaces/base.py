"""
Space normalization module for aligning estimator outputs and feature space.

This module defines the normalization layer used to project raw inputs
and model outputs into a consistent representation space before
distance computation and kernel processing.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
In COBRA-style ensembles, different estimators may produce outputs
on different scales or distributions. Similarly, feature spaces may
not be directly comparable across models.

The space normalizer ensures:

- consistent scaling between estimators
- alignment between input features and model outputs
- stable distance computation
- improved kernel weighting behavior

This step acts as a bridge between:

- estimator outputs
- distance computation space

Design goals
------------
- unify scaling and transformation logic
- support multiple normalization strategies
- ensure compatibility across heterogeneous estimators
- enable plug-and-play preprocessing pipelines
- maintain factory-based extensibility

Key idea
--------
Normalization ensures that:

    X (input space)
    model_outputs (prediction space)

are brought into a shared representation space where distance
functions become meaningful and comparable.

Examples of transformations include:

- standardization (z-score scaling)
- min-max scaling
- joint embedding projection
- output calibration alignment

Examples
--------
>>> @SpaceNormalizerFactory.register("standard")
... class StandardNormalizer(BaseSpaceNormalizer):
...     def transform(self, X, model_outputs):
...         X = (X - X.mean()) / X.std()
...         model_outputs = (model_outputs - model_outputs.mean()) / model_outputs.std()
...         return X, model_outputs
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cobra.core.factory import BaseFactory


class BaseSpaceNormalizer(ABC):
    """
    Abstract base class for space normalization strategies.

    This class defines the interface for transforming raw inputs
    and model outputs into a unified representation space.

    Pipeline role
    -------------
    Normalizers ensure compatibility between:

    - estimator predictions
    - feature representations
    - distance computations
    - kernel weighting functions

    Notes
    -----
    Subclasses must implement ``transform``.

    The transformation typically modifies both X and model outputs
    to ensure they live in a comparable metric space.

    Examples
    --------
    >>> class IdentityNormalizer(BaseSpaceNormalizer):
    ...     def transform(self, X, model_outputs):
    ...         return X, model_outputs
    """

    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        model_outputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform input and model output into a shared space.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        model_outputs : np.ndarray
            Predictions from estimators.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Transformed (X, model_outputs) in normalized space.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> Xn, Yn = normalizer.transform(X, preds)
        """
        raise NotImplementedError


class SpaceNormalizerFactory(BaseFactory):
    """
    Factory for space normalization strategies.

    This registry-based factory enables dynamic selection of
    normalization methods used to align estimator outputs and
    input feature spaces.

    It is commonly used in:

    - multi-model ensemble alignment
    - COBRA distance calibration pipelines
    - heterogeneous estimator fusion
    - preprocessing configuration systems

    Examples
    --------
    >>> normalizer = SpaceNormalizerFactory.create("standard")

    >>> Xn, Yn = normalizer.transform(X, predictions)
    """
    pass
