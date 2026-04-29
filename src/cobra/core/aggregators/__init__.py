"""
Aggregation package for final COBRA consensus outputs.

This package provides the final aggregation stage of the COBRA pipeline,
where selected neighbor targets and optional kernel weights are combined
into a single prediction.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
After the kernel stage identifies relevant neighbors and computes
their associated influence weights, the aggregation stage produces
the final consensus prediction.

This is the last computational step before returning model output.

Main responsibilities
---------------------

- combine neighbor target values
- apply optional kernel weights
- support regression and classification tasks
- provide robust consensus strategies

Available components
--------------------

Base classes
^^^^^^^^^^^^

- ``BaseAggregator``
    Abstract interface for all aggregation strategies.

- ``AggregatorFactory``
    Registry-based factory for dynamic aggregator creation.

Built-in implementations
^^^^^^^^^^^^^^^^^^^^^^^^

- ``SimpleMeanAggregator``
    Arithmetic mean of candidate values.

- ``WeightedMeanAggregator``
    Weighted mean using kernel-generated weights.

- ``MajorityVoteAggregator``
    Most frequent class label for classification tasks.

Examples
--------
>>> from cobra.core.aggregation import AggregatorFactory

>>> aggregator = AggregatorFactory.create("weighted_mean")

>>> prediction = aggregator.aggregate(
...     values=[1.2, 1.5, 1.8],
...     weights=[0.2, 0.5, 0.3]
... )

Exports
-------
This package exposes the most commonly used aggregation classes
directly for convenient imports.
"""

from .base import (
    BaseAggregator,
    AggregatorFactory,
)

from .builtin import (
    MajorityVoteAggregator,
    SimpleMeanAggregator,
    WeightedMeanAggregator,
)

__all__ = [
    "BaseAggregator",
    "AggregatorFactory",
    "MajorityVoteAggregator",
    "SimpleMeanAggregator",
    "WeightedMeanAggregator",
]
