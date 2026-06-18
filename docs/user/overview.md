# User Overview

This section explains how to use `kfc-procedure` without reading the source code.

## Goal

KFCProcedure helps train models on heterogeneous data. It assumes that one global model may not be ideal, so it creates a clusterwise learning pipeline:

| Step | Name | Meaning |
|---|---|---|
| K-step | Clustering | Split observations into similar groups using Bregman divergences |
| F-step | Fitting | Train local supervised models in each cluster |
| C-step | Combining | Combine divergence-level predictions into one final prediction |

## When to use it

Use KFCProcedure when your dataset may contain hidden subgroups, one model underfits different regions of the data, or you want a local-model ensemble with a scikit-learn-style interface.

!!! warning "Small datasets"
    Avoid large `n_clusters` when the dataset is small. Some clusters may have too few samples for stable local training.

## Safe first choices

| Task | Estimator | Divergence | Local model | Combiner |
|---|---|---|---|---|
| Regression | `KFCRegressor` | `euclidean` | `linear_regression` | `mean` |
| Classification | `KFCClassifier` | `euclidean` | `decision_tree_classifier` | `majority_vote` |

## Main registered components

| Component type | Available values |
|---|---|
| Divergences | `euclidean`, `gkl`, `is`, `logistic` |
| Regression combiners | `mean`, `weighted_mean`, `stacking_regressor`, `gradientcobra`, `mixcobra` |
| Classification combiners | `majority_vote`, `stacking_classifier`, `combined_classifier` |
| Local models | many sklearn estimators auto-registered in snake_case |
