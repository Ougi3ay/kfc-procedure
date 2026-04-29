"""
COBRA core module.

This package contains the fundamental building blocks of the COBRA
and MIXCOBRA pipeline architecture.

Pipeline overview
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
The core module provides all reusable components required to build
end-to-end consensus learning systems.

It implements a fully modular, factory-driven architecture where each
stage of the pipeline can be independently replaced or extended.

Main design goals:

- modular pipeline construction
- consistent factory-based instantiation
- extensible research-friendly architecture
- clear separation of concerns between stages
- support for COBRA / GradientCOBRA / MIXCOBRA variants

Core components
---------------

Adapters
^^^^^^^^
Transform raw distance matrices using learnable or fixed parameters.

Aggregators
^^^^^^^^^^^^
Combine neighbor predictions into a final consensus output.

Distances
^^^^^^^^^
Compute pairwise distances between samples in feature space.

Estimators
^^^^^^^^^^^
Base models used as experts in the prediction pool.

Kernels
^^^^^^^
Transform distances into similarity / influence weights.

Losses
^^^^^^
Evaluate prediction error for optimization objectives.

Optimizers
^^^^^^^^^^
Search or refine model parameters (gradient-based or discrete).

Spaces
^^^^^^
Normalize and align estimator outputs into a shared representation.

Splitters
^^^^^^^^^
Partition data into training and calibration subsets.

Factory system
--------------

- ``BaseFactory``
    Generic registry-based factory used across all components.

Design philosophy
-----------------
Each module is designed to be:

- independently replaceable
- configurable via string-based factory registration
- compatible with black-box optimization workflows
- suitable for ensemble learning research and experimentation

Example usage
-------------
>>> from cobra.core.splitters import SplitterFactory
>>> from cobra.core.estimators import EstimatorFactory
>>> from cobra.core.optimizers import GradientOptimizerFactory

>>> splitter = SplitterFactory.create("holdout")
>>> estimator = EstimatorFactory.create("ridge")
>>> optimizer = GradientOptimizerFactory.create("gradient_descent")
"""

from .adapters import *
from .aggregators import *
from .distances import *
from .estimators import *
from .kernels import *
from .losses import *
from .optimizers import *
from .spaces import *
from .splitters import *

from .factory import BaseFactory

__all__ = [
    "BaseFactory",
]