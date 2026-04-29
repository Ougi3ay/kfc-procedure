"""
Built-in space normalization strategies for COBRA pipelines.

This module provides concrete implementations of space normalization
used to align input features and estimator outputs before distance
computation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Space normalization ensures that different representations produced
by estimators and raw input features are comparable in a shared
metric space.

Without normalization, COBRA-style pipelines may suffer from:

- inconsistent scales between models
- biased distance computations
- unstable kernel weighting
- poor ensemble calibration

This stage ensures that both:

- input features (X)
- model predictions (model_outputs)

are properly scaled before distance computation.

Normalization strategies
------------------------

1. IdentitySpaceNormalizer

    No transformation is applied.

2. GradientCOBRASpaceNormalizer

    Normalizes model outputs using a shared normalization constant.

3. MixCOBRASpaceNormalizer

    Separately normalizes input features and model outputs.

Design goals
------------
- enforce consistent scaling across heterogeneous models
- support configurable normalization constants
- allow per-component scaling strategies (X vs Y)
- integrate with COBRA distance and kernel stages
- remain factory-driven for experiment flexibility

Examples
--------
>>> normalizer = SpaceNormalizerFactory.create("mixcobra")
>>> Xn, Yn = normalizer.transform(X, model_outputs)
"""

from __future__ import annotations

from cobra.core.spaces.base import BaseSpaceNormalizer, SpaceNormalizerFactory
from cobra.utils.preprocessing import compute_normalization_constant


@SpaceNormalizerFactory.register("identity")
class IdentitySpaceNormalizer(BaseSpaceNormalizer):
    """
    Identity space normalizer.

    This normalizer performs no transformation and returns inputs
    unchanged.

    Use cases
    ---------
    - debugging pipelines
    - baseline comparisons
    - ensuring raw-space behavior

    Examples
    --------
    >>> Xn, Yn = IdentitySpaceNormalizer().transform(X, Y)
    """

    def transform(self, X, model_outputs):
        return X, model_outputs


@SpaceNormalizerFactory.register("gradientcobra")
class GradientCOBRASpaceNormalizer(BaseSpaceNormalizer):
    """
    Space normalizer for GradientCOBRA pipeline.

    This normalizer scales model outputs using a shared normalization
    constant computed from predictions.

    Parameters
    ----------
    norm_constant : optional
        Predefined normalization constant. If None, it is computed
        dynamically.

    Notes
    -----
    This ensures stable scaling of model outputs before distance
    computation in gradient-based COBRA variants.

    Examples
    --------
    >>> normalizer = GradientCOBRASpaceNormalizer()
    >>> Xn, Yn = normalizer.transform(X, preds)
    """

    def __init__(self, norm_constant=None):
        self.norm_constant = norm_constant

    def transform(self, X, model_outputs):
        M = model_outputs.shape[1]

        normalize_constant = compute_normalization_constant(
            model_outputs,
            self.norm_constant,
            scale_factor=30.0,
            M=M,
        )

        Y = model_outputs / normalize_constant

        return X, Y


@SpaceNormalizerFactory.register("mixcobra")
class MixCOBRASpaceNormalizer(BaseSpaceNormalizer):
    """
    Space normalizer for MixCOBRA pipeline.

    This normalizer independently scales input features and model
    outputs using separate normalization constants.

    Parameters
    ----------
    norm_constant_x : optional
        Normalization constant for input features.

    norm_constant_y : optional
        Normalization constant for model outputs.

    Notes
    -----
    This approach is useful when X and Y come from different
    distributions and require independent scaling.

    Examples
    --------
    >>> normalizer = MixCOBRASpaceNormalizer()
    >>> Xn, Yn = normalizer.transform(X, preds)
    """

    def __init__(self, norm_constant_x=None, norm_constant_y=None):
        self.norm_constant_x = norm_constant_x
        self.norm_constant_y = norm_constant_y

    def transform(self, X, model_outputs):
        M = model_outputs.shape[1]

        normalize_constant_x = compute_normalization_constant(
            X,
            self.norm_constant_x,
            scale_factor=30.0,
            M=M,
        )

        normalize_constant_y = compute_normalization_constant(
            model_outputs,
            self.norm_constant_y,
            scale_factor=30.0,
            M=M,
        )

        X = X / normalize_constant_x
        Y = model_outputs / normalize_constant_y

        return X, Y
