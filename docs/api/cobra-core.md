# COBRA Core API


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


The COBRA subsystem separates model aggregation into smaller, reusable components.

## Component responsibilities

| Component family | Base class | Factory | Responsibility |
|---|---|---|---|
| Estimators | `BaseEstimator` | `EstimatorFactory` | Fit and predict base model outputs. |
| Splitters | `BaseDataSplitter` | `SplitterFactory` | Create training/calibration splits. |
| Normalizers | `BaseNormalizer` | `NormalizerFactory` | Standardize or min-max scale arrays. |
| Distances | `BaseDistance` | `DistanceFactory` | Compute pairwise distance matrices. |
| Kernel adapters | `BaseKernelAdapter` | `KernelAdapterFactory` | Transform distances with tunable parameters. |
| Kernels | `BaseKernel` | `KernelFactory` | Convert distances to weights. |
| Losses | `BaseLoss` | `LossFactory` | Evaluate objective functions. |
| Cross-validation | `BaseCrossValidator` | `CVFactory` | Produce folds for optimization. |
| Optimizers | `BaseOptimizer` | `OptimizerFactory` | Search or optimize kernel parameters. |
| Aggregators | `BaseAggregator` | `AggregatorFactory` | Aggregate weighted calibration targets. |

## Distances

Registered distance names include `euclidean`, `manhattan`, `hamming`, `cosine`, and `minkowski`. Aliases such as `l1`, `l2`, and `lp` are also registered.

## Kernels

Registered kernels include `rbf`, `gaussian`, `radial`, `cobra`, `naive`, `triangular`, `epanechnikov`, `biweight`, `triweight`, `exponential`, `cauchy`, and `reverse_cosh`.

## Optimizers

| Optimizer | Factory name | Notes |
|---|---|---|
| `GridSearchOptimizer` | `grid` | Enumerates parameter grid candidates. |
| `GradientDescentOptimizer` | `gd` | Uses numerical gradients. |
| `MomentumOptimizer` | `momentum` | Gradient descent with momentum. |
| `AdamOptimizer` | `adam` | Adam-style adaptive updates. |

## Aggregators

`WeightedMeanAggregator` is used for regression-style aggregation. `WeightedVoteAggregator` is used for classification-style voting and probability aggregation.

## Splitters

`RandomHoldoutSplitter` creates disjoint training/evaluation splits. `OverlapSplitter` creates splits with configurable overlap, used by COBRA-style workflows that need shared calibration regions.
