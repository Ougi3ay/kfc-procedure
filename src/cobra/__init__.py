"""
COBRA package

High-level package documentation for the COBRA family of models (COBRA,
GradientCOBRA, MixCOBRA). This package implements a modular, factory-driven
architecture for consensus-based regression and classification.

Pipeline summary
---------------
Input -> Splitter -> Estimators -> Normalize Constants / Space Normalizer
-> Distance -> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
This package provides reusable components for building ensemble consensus
models that combine predictions from an expert pool using distance-based
similarity and kernel-weighted aggregation. Components are instantiated via
registry-based factories to keep the system extensible and configuration
driven.

Use cases
---------
- Research: compare aggregation strategies, adapters, and optimizers
- Production: pluggable pipelines for robust ensemble regression/classification
- Education: clear separation of pipeline stages for teaching

Compatibility
-------------
Modules are designed to be compatible with scikit-learn estimator APIs
(``fit``/``predict``) and to work with documentation tools such as Sphinx.
"""

from .mixcobra import MixCOBRARegressor
from .gradientcobra import GradientCOBRA
from .combine_classifier import CombineClassifier
from .superlearner import SuperLearner

__all__ = [
    "MixCOBRARegressor",
    "GradientCOBRA",
    "CombineClassifier",
    "SuperLearner",
]
