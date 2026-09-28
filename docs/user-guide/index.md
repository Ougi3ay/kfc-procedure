# User Guide

The **User Guide** explains how to configure, train, customize, and inspect the
estimators provided by `kfc-procedure`.

If you are new to the library, start with the KFC Procedure guides. If you
already understand the main algorithms, use the sections below to configure
clustering, estimators, consensus methods, kernels, distances, optimization,
and advanced workflows.

---

## Start here

<div class="grid cards" markdown>

-   :material-play-circle-outline:{ .lg .middle } **KFC Procedure**

    ---

    Learn how to fit the full **K-means → Fit → Consensus** workflow.

    [:octicons-arrow-right-24: Fitting KFC](kfc/fitting-kfc.md)

-   :material-chart-line:{ .lg .middle } **Regression**

    ---

    Use KFC for continuous targets and configure regression combiners.

    [:octicons-arrow-right-24: KFC Regression](kfc/regression.md)

-   :material-shape-outline:{ .lg .middle } **Classification**

    ---

    Use KFC for class labels and configure classification consensus methods.

    [:octicons-arrow-right-24: KFC Classification](kfc/classification.md)

-   :material-source-merge:{ .lg .middle } **Consensus methods**

    ---

    Compare mean, weighted mean, stacking, GradientCOBRA, MixCOBRA,
    majority vote, and CombinedClassifier.

    [:octicons-arrow-right-24: Consensus Overview](consensus/overview.md)

</div>

---

## KFC Procedure

The KFC workflow is organized into three stages:

\[
\boxed{
\text{K-means}
\rightarrow
\text{Fit}
\rightarrow
\text{Consensus}
}
\]

The corresponding implementation can be configured stage by stage.

<div class="grid cards" markdown>

-   :material-numeric-1-circle:{ .lg .middle } **K-Step**

    ---

    Configure clustering and Bregman divergences.

    [:octicons-arrow-right-24: Configure K-Step](kfc/configuring-k-step.md)

-   :material-numeric-2-circle:{ .lg .middle } **F-Step**

    ---

    Configure the local predictive models fitted inside each cluster.

    [:octicons-arrow-right-24: Configure F-Step](kfc/configuring-f-step.md)

-   :material-numeric-3-circle:{ .lg .middle } **C-Step**

    ---

    Configure the final consensus or aggregation method.

    [:octicons-arrow-right-24: Configure C-Step](kfc/configuring-c-step.md)

-   :material-magnify-expand:{ .lg .middle } **Inspect fitted models**

    ---

    Explore fitted clusters, local estimators, predictions, and aggregation
    state.

    [:octicons-arrow-right-24: Inspect Fitted Models](kfc/inspecting-fitted-models.md)

</div>

### Main KFC guides

- [Fitting KFC](kfc/fitting-kfc.md)
- [Regression](kfc/regression.md)
- [Classification](kfc/classification.md)
- [Configure K-Step](kfc/configuring-k-step.md)
- [Configure F-Step](kfc/configuring-f-step.md)
- [Configure C-Step](kfc/configuring-c-step.md)
- [Inspect Fitted Models](kfc/inspecting-fitted-models.md)

---

## Clustering

The K-Step creates candidate partitions of the input data.

KFC can use several **Bregman divergences** so that the same dataset is viewed
through different clustering geometries.

Start with:

- [Bregman Divergences](clustering/bregman-divergences.md)

Available divergence families include:

```text
Squared Euclidean
Generalized Kullback–Leibler
Logistic
Itakura–Saito
```

!!! note

    Some divergences have domain requirements. For example, some require
    positive-valued inputs or values inside a bounded interval.

    Review the divergence guide before applying a divergence to raw data.

---

## Base estimators

The F-Step fits predictive models inside the clusters constructed during the
K-Step.

The package also uses estimator pools directly inside COBRA-style aggregation
methods.

Use these guides to configure them:

- [Base Estimators](estimators/base-estimators.md)
- [Estimator Parameters](estimators/estimator-parameters.md)
- [Custom Estimators](estimators/custom-estimators.md)

Typical regression estimators include:

```text
linear_regression
ridge
ridge_cv
lasso
lasso_cv
k_neighbors_regressor
random_forest_regressor
svr
```

Typical classification estimators include:

```text
logistic_regression
decision_tree_classifier
random_forest_classifier
svc
k_neighbors_classifier
```

---

## Consensus methods

The C-Step combines the candidate predictions into one final prediction.

Different consensus methods are available depending on the task.

### Regression

Common regression combiners include:

```text
mean
weighted_mean
stacking
gradientcobra
mixcobra
```

See:

- [Consensus Overview](consensus/overview.md)
- [Regression Combiners](consensus/regression-combiners.md)
- [Weighted Mean](consensus/weighted-mean.md)
- [Stacking](consensus/stacking.md)

For detailed algorithm explanations, see the concept pages:

- [GradientCOBRA](../getting-started/concepts/gradientcobra.md)
- [MixCOBRA](../getting-started/concepts/mixcobra.md)

### Classification

Common classification combiners include:

```text
majority_vote
stacking
combined_classifier
```

See:

- [Classification Combiners](consensus/classification-combiners.md)
- [Majority Vote](consensus/majority-vote.md)
- [Stacking](consensus/stacking.md)

For the prediction-space consensus method, see:

- [CombinedClassifier](../getting-started/concepts/combined-classifier.md)

---

## COBRA components

`GradientCOBRA`, `MixCOBRA`, and `CombinedClassifier` share several reusable
components.

Understanding these components makes it easier to customize the aggregation
behavior.

<div class="grid cards" markdown>

-   :material-vector-polyline:{ .lg .middle } **Prediction space**

    ---

    Learn how multiple model outputs become features for aggregation.

    [:octicons-arrow-right-24: Prediction Space](cobra/prediction-space.md)

-   :material-ruler:{ .lg .middle } **Distances**

    ---

    Configure how observations or prediction vectors are compared.

    [:octicons-arrow-right-24: Distances](cobra/distances.md)

-   :material-chart-bell-curve:{ .lg .middle } **Kernels**

    ---

    Transform distance into similarity weights.

    [:octicons-arrow-right-24: Kernels](cobra/kernels.md)

-   :material-sigma:{ .lg .middle } **Aggregators**

    ---

    Control how weighted targets or labels are combined.

    [:octicons-arrow-right-24: Aggregators](cobra/aggregators.md)

</div>

### Available COBRA guides

- [Prediction Space](cobra/prediction-space.md)
- [Data Splitting](cobra/data-splitting.md)
- [Distances](cobra/distances.md)
- [Kernels](cobra/kernels.md)
- [Aggregators](cobra/aggregators.md)
- [Cross-validation](cobra/cross-validation.md)
- [Loss Functions](cobra/losses.md)
- [Optimization](cobra/optimization.md)
- [Gradient Optimization](cobra/gradient-optimization.md)
- [Normalization](cobra/normalization.md)

---

## Distances

The aggregation methods can compare observations using several distance
functions.

Typical options include:

| Name | Alias | Typical use |
| --- | --- | --- |
| `euclidean` | `l2` | Continuous prediction or feature vectors |
| `manhattan` | `l1` | Absolute-coordinate differences |
| `minkowski` | `lp` | Generalized \(L_p\) distance |
| `cosine` | — | Direction-based similarity |
| `hamming` | — | Discrete classifier prediction vectors |

For example:

```python
model = GradientCOBRA(
    distance="euclidean",
)
```

or:

```python
model = CombinedClassifier(
    distance="hamming",
)
```

See [Distances](cobra/distances.md) for details.

---

## Kernels

Distances are converted into similarity weights using a kernel.

Available kernel families include:

```text
rbf
gaussian
radial
cobra
naive
epanechnikov
triangular
biweight
triweight
cauchy
exponential
reverse_cosh
```

For example:

```python
model = GradientCOBRA(
    kernel="rbf",
)
```

or:

```python
model = MixCOBRARegressor(
    kernel="epanechnikov",
)
```

See [Kernels](cobra/kernels.md).

---

## Cross-validation and losses

Aggregation hyperparameters are usually tuned on an aggregation subset using
cross-validation.

Available strategies include:

```text
K-fold
stratified K-fold
time-series split
```

Use:

- [Cross-validation](cobra/cross-validation.md)
- [Loss Functions](cobra/losses.md)

Typical losses include:

```text
mse
mae
huber
quantile
log_loss
hinge
```

---

## Optimization

Several aggregation methods optimize a bandwidth or other kernel parameters.

Two major optimization styles are available:

### Grid search

Evaluate a predefined list of candidate values.

```python
model = GradientCOBRA(
    bandwidth_list=[0.01, 0.1, 0.5, 1.0, 2.0],
    optimizer="grid",
    opt_method="grid",
)
```

### Gradient-based optimization

Use iterative optimization for smooth objectives and compatible kernels.

```python
model = GradientCOBRA(
    kernel="rbf",
    optimizer="adam",
    opt_method="grad",
    learning_rate=0.05,
)
```

See:

- [Optimization](cobra/optimization.md)
- [Gradient Optimization](cobra/gradient-optimization.md)

---

## Advanced workflows

The library also supports more specialized workflows.

### Precomputed predictions

You can provide a matrix whose columns already contain predictions from
external models.

```python
model.fit(
    prediction_matrix,
    y,
    as_predictions=True,
)
```

See [Precomputed Predictions](advanced/precomputed-predictions.md).

### Custom aggregation data

Instead of allowing the estimator to split the data automatically, provide the
estimator-training and aggregation sets directly.

```python
model.fit(
    X_k,
    y_k,
    X_l=X_l,
    y_l=y_l,
)
```

See [Custom Aggregation Data](advanced/custom-aggregation-data.md).

### Parallel execution

Some estimator-pool operations support `n_jobs`.

```python
model = GradientCOBRA(
    n_jobs=-1,
)
```

See [Parallel Execution](advanced/parallel-execution.md).

### Reproducibility

Control random splits and compatible estimators with `random_state`.

```python
model = GradientCOBRA(
    random_state=42,
)
```

See [Reproducibility](advanced/reproducibility.md).

### Extending the package

The package uses registries and factories for several component families.

You can extend the library with custom:

```text
estimators
divergences
distances
kernels
aggregators
loss functions
optimizers
combiners
```

See [Extending KFC Procedure](advanced/extending-kfc-procedure.md).

---

## Concepts vs User Guide

The documentation is divided into two complementary parts.

### Concepts

Use the **Concepts** section when you want to understand how an algorithm works.

- [KFC Procedure](../getting-started/concepts/kfc-procedure.md)
- [CombinedClassifier](../getting-started/concepts/combined-classifier.md)
- [MixCOBRA](../getting-started/concepts/mixcobra.md)
- [GradientCOBRA](../getting-started/concepts/gradientcobra.md)

### User Guide

Use the **User Guide** when you want to:

- configure an estimator,
- choose components,
- change kernels or distances,
- tune optimization,
- provide custom data splits,
- inspect a fitted estimator,
- or extend the package.

!!! tip "Recommended path"

    If this is your first time using `kfc-procedure`:

    1. Read the [KFC Procedure concept](../getting-started/concepts/kfc-procedure.md).
    2. Continue with [Fitting KFC](kfc/fitting-kfc.md).
    3. Read either [Regression](kfc/regression.md) or
       [Classification](kfc/classification.md).
    4. Use the COBRA component guides when you need more control over the
       consensus stage.

---

## Troubleshooting

If a configuration does not behave as expected, see:

[:octicons-arrow-right-24: Troubleshooting](troubleshooting.md)

Common topics include:

```text
invalid divergence domains
unknown estimator names
invalid kernel or distance names
empty or zero-weight neighborhoods
incompatible optimizer / kernel combinations
aggregation-data shape mismatches
precomputed prediction shape mismatches
reproducibility
```