"""
Concrete dataset splitters for COBRA-style training and calibration workflows.

This module implements common dataset partitioning strategies used in
COBRA pipelines, including holdout, K-fold cross-validation, and
overlapping splits.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Splitters define how input data is partitioned into subsets used for:

- estimator training
- calibration / aggregation
- cross-validation
- ensemble evaluation

In COBRA-style systems, splitting is index-based to ensure:

- consistent alignment across estimators
- reproducible experimental design
- compatibility with ensemble aggregation logic

Available strategies
--------------------

1. RandomHoldoutSplitter
^^^^^^^^^^^^^^^^^^^^^^^^

Randomly partitions data into training and calibration sets.

Typical use cases:
- simple train/calibration split
- baseline COBRA experiments

2. KFoldSplitter
^^^^^^^^^^^^^^^^

Generates K-fold cross-validation splits.

Typical use cases:
- model evaluation
- robust performance estimation
- ensemble validation

3. OverlapSplitter
^^^^^^^^^^^^^^^^^^

Creates overlapping training and calibration sets.

Typical use cases:
- soft calibration setups
- robustness experiments
- MIXCOBRA-style overlap learning

Design goals
------------
- index-preserving splitting (no data duplication)
- reproducible randomness via seeds
- flexible split strategies
- compatibility with ensemble pipelines
- support for advanced overlap calibration

Examples
--------
>>> splitter = SplitterFactory.create("holdout")
>>> train_idx, cal_idx = splitter.split(X, y)

>>> splitter = SplitterFactory.create("kfold")
>>> folds = splitter.split(X, y)

>>> splitter = SplitterFactory.create("split_overlap")
>>> train_idx, agg_idx = splitter.split(X, y)
"""

from __future__ import annotations

import numpy as np

from cobra.core.validators.base import BaseCrossValidator, CVFactory


@CVFactory.register("kfold")
class KFoldCV(BaseCrossValidator):

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x, y):
        n_samples = len(x)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)

        for i in range(self.n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate(
                [folds[j] for j in range(self.n_splits) if j != i]
            )
            yield train_idx, val_idx
