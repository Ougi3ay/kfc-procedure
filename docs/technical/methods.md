# Methods

This page explains the main methods implemented in the codebase, why they are used, and possible alternatives.

---

## 1. Bregman divergence-based clustering

### What it does

The K-step uses `BregmanKMeans`, a generalization of classical K-Means. Instead of using only squared Euclidean distance, it supports multiple Bregman divergences:

| Identifier | Class | Domain | Typical data type |
|---|---|---|---|
| `euclidean` | `SquaredEuclidean` | all real values | generic numerical features |
| `gkl` | `GKLDivergence` | strictly positive values | positive count-like data |
| `is` | `ItakuraSaito` | strictly positive values | scale/spectral positive data |
| `logistic` | `LogisticLoss` | values in `(0, 1)` | probability-like features |

### Why it is used

Different divergences imply different geometric assumptions. For heterogeneous data, one distance measure may not capture all useful structure. By fitting one clustering model per divergence, KFCProcedure creates multiple views of the same dataset.

### Alternatives

| Alternative | When to use | Trade-off |
|---|---|---|
| Standard K-Means | General numeric data | Simpler but only Euclidean geometry. |
| Gaussian Mixture Models | Soft probabilistic clusters | More assumptions, can be sensitive to covariance estimation. |
| Spectral clustering | Non-convex cluster shapes | Higher computational cost. |
| Hierarchical clustering | Small datasets and interpretability | Usually expensive for large `n`. |
| Density-based clustering | irregular clusters/noise | Harder to integrate with fixed local models. |

---

## 2. Multi-divergence K-step

### What it does

`KStep` accepts a list of divergences and fits an independent `BregmanKMeans` model for each divergence.

Output:

```python
clusters_ = {
    "euclidean": array([...]),
    "gkl": array([...]),
    ...
}
```

### Why it is used

Each divergence produces a different partition. Downstream local models can specialize under each partition, and the C-step can combine the resulting predictions.

### Alternatives

- Use only one divergence for speed and simplicity.
- Use model selection to choose the best divergence.
- Use ensemble clustering or consensus clustering before local modeling.

---

## 3. Cluster-local supervised learning

### What it does

`FStep` trains a separate local supervised model inside each cluster for each divergence.

For divergence `d` and cluster `k`, it trains:

```text
model[d][k].fit(X_cluster, y_cluster)
```

At prediction time, each sample is assigned to a cluster for each divergence, then predicted by the corresponding local model.

### Why it is used

Local models can capture different predictive behavior in different data regions. This can reduce bias when the relationship between `X` and `y` is not globally uniform.

### Alternatives

| Alternative | Description |
|---|---|
| Single global model | Simpler and more stable for small datasets. |
| Mixture of experts | Learns soft gating instead of hard clustering. |
| Random forest / boosting | Implicitly creates local partitions but less explicit. |
| Local regression / nearest neighbors | Uses local neighborhoods without learned clusters. |

---

## 4. Prediction matrix construction

After the F-step, the output is a prediction matrix:

```text
P ∈ R^{n_samples × n_divergences}
```

Each column is the prediction produced under one divergence-specific local modeling system.

Example with three divergences:

| sample | euclidean prediction | gkl prediction | logistic prediction |
|---:|---:|---:|---:|
| 1 | 10.2 | 9.8 | 10.5 |
| 2 | 4.1 | 5.0 | 4.8 |
| 3 | 7.7 | 7.4 | 7.9 |

The C-step receives this matrix and learns or applies an aggregation rule.

---

## 5. C-step combiners

### Regression combiners

| Identifier | Implementation | Method |
|---|---|---|
| `mean` | `MeanCombiner` | Row-wise arithmetic mean. |
| `weighted_mean` | `WeightedMeanCombiner` | Linear regression over prediction columns. |
| `stacking_regressor` | `StackingRegressorCombiner` | Meta-regressor over predictions. |
| `gradientcobra` | `GradientCOBRACombiner` | COBRA kernel aggregation over prediction matrix. |
| `mixcobra` | `MixCOBRACombiner` | Mixed input/prediction-space COBRA aggregation. |

### Classification combiners

| Identifier | Implementation | Method |
|---|---|---|
| `majority_vote` | `MajorityVoteCombiner` | Hard vote over predicted labels. |
| `stacking_classifier` | `StackingClassifierCombiner` | Logistic-regression meta-classifier. |
| `combined_classifier` | `CobraClassifierCombiner` | COBRA-based classifier aggregation. |

### Why they are used

The combiner acts as the final fusion layer. Simple combiners reduce variance and are easy to interpret. Learned combiners can adaptively weight the divergence views.

### Alternatives

- Use cross-validation to select the best single divergence.
- Use Bayesian model averaging.
- Use a neural gating network.
- Use calibration models for probabilistic classification.

---

## 6. COBRA-style ensemble aggregation

### What it does

COBRA methods operate in a **prediction space** formed by base estimator outputs. Samples are considered similar if base estimators produce similar predictions for them.

The package implements:

| Estimator | Task | Key idea |
|---|---|---|
| `GradientCOBRA` | regression | Optimize one bandwidth for kernel-weighted aggregation in prediction space. |
| `MixCOBRARegressor` | regression | Learn distance mixing between original input space and prediction space. |
| `CombinedClassifier` | classification | Kernel-weighted voting/probability aggregation in prediction space. |
| `SuperLearner` | regression | Cross-validated stacking with base learners and meta learners. |

### Why it is used

COBRA aggregation can be robust when individual base estimators are imperfect but their agreement patterns are informative. It uses neighborhoods in prediction space rather than only raw input space.

### Alternatives

| Alternative | Difference |
|---|---|
| Bagging | Averages many models trained on bootstrap samples. |
| Boosting | Sequentially corrects previous errors. |
| Stacking | Learns a meta-model over predictions. |
| KNN regression/classification | Aggregates in input space only. |
| Random forest | Tree-based implicit ensemble; less modular for arbitrary predictions. |

---

## 7. Registry-based component resolution

### What it does

The code uses factory registries to map strings to classes. For example:

```python
BregmanDivergenceFactory.create("euclidean")
CombinerFactory.create("weighted_mean")
DistanceFactory.create("cosine")
```

### Why it is used

This design makes the package extensible. New components can be registered without changing the central pipeline code.

### Alternatives

- Hard-coded `if/elif` component selection.
- Configuration files with import paths.
- Python entry points for external plugins.

---

## 8. Internal calibration split

`KFCProcedure.fit` splits the input data into two halves:

- `X_k`, `y_k`: used for K-step clustering and F-step local model training;
- `X_l`, `y_l`: used for C-step combiner fitting.

This reduces overfitting in the aggregation layer because the combiner is trained on predictions for samples not used to train the local models.

!!! warning "Current implementation detail"
    The KFC internal split ratio is fixed at `test_size=0.5` in `KFCProcedure.fit`. There is no public `split_ratio` argument in the current wrapper, unlike the COBRA estimators.
