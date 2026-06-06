"""
KFC pipeline core stages.

This package defines the three main stages of the KFCProcedure
framework:

    K-step → F-step → C-step

These stages implement a modular ensemble learning pipeline based on
Bregman divergences and cluster-wise model specialization.

Pipeline overview
-----------------
The KFC pipeline follows a three-stage decomposition:

1. K-step (Clustering stage)
   - Fits multiple BregmanKMeans models under different divergences
   - Produces divergence-specific cluster assignments

2. F-step (Local model stage)
   - Trains a local predictive model per cluster
   - Produces cluster-aware predictions for each divergence

3. C-step (Aggregation stage)
   - Combines divergence-specific predictions
   - Produces final ensemble output using a combiner strategy

Design philosophy
------------------
The framework is designed around the idea of:

    "Different divergences define different geometries,
     and each geometry induces a different learning bias."

This allows the pipeline to:

- Capture heterogeneous structure in data
- Combine multiple metric-induced hypotheses
- Improve robustness over single-metric models

Modules
-------
KStep
    Divergence-aware clustering stage.

FStep
    Cluster-wise local model fitting stage.

CStep
    Final aggregation layer for ensemble prediction.

Notes
-----
Each stage is compatible with scikit-learn-like APIs and is designed
to be composable in pipeline-style workflows.

Typical usage:

    k = KStep(...)
    f = FStep(...)
    c = CStep(...)

    clusters = k.fit_predict(X)
    preds = f.fit(X, y, clusters)
    final = c.fit_predict(preds, y)
"""

from __future__ import annotations
from .kstep import KStep
from .fstep import FStep
from .cstep import CStep

__all__ = [
    "KStep",
    "FStep",
    "CStep",
]
