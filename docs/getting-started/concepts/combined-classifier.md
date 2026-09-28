# CombinedClassifier

`CombinedClassifier` is an ensemble classifier that combines several base
classification models using **similarity in prediction space**.

Instead of selecting a single classifier, it asks a different question:

> Which training observations receive predictions similar to this new
> observation from the collection of base classifiers?

Observations with similar prediction patterns receive more influence in the
final classification.

This idea is based on the combined-classifier approach introduced by
Mojirsheibani, where observations are grouped according to the predictions
produced by multiple classifiers and the final label is obtained from the
labels of observations belonging to the same prediction region.

The implementation in `kfc-procedure` extends this idea with:

- configurable base classifiers,
- prediction-space distance functions,
- kernel-based similarity weights,
- weighted class voting,
- cross-validation,
- automatic bandwidth optimization.

---

## Why combine classifiers?

Different classifiers learn different structures from the same dataset.

For example:

- logistic regression captures approximately linear decision boundaries,
- decision trees capture rule-based nonlinear relationships,
- support vector machines can construct more flexible boundaries,
- k-nearest neighbors makes decisions from local neighborhoods.

There is no guarantee that one classifier will work best for every region of
the input space.

`CombinedClassifier` therefore does not simply select one model.

It constructs a new representation from the predictions of several models and
performs the final classification in that **prediction space**.

---

## Prediction space

Assume that we train \(M\) classifiers

\[
C_1, C_2, \ldots, C_M.
\]

For an observation \(x\), define its prediction vector as

\[
\mathbf{C}(x)
=
\left(
C_1(x),
C_2(x),
\ldots,
C_M(x)
\right).
\]

For example, suppose four classifiers produce

```text
Logistic Regression -> 1
Decision Tree       -> 0
SVC                 -> 1
KNN                 -> 1
```

Then the prediction-space representation of the observation is

\[
\mathbf{C}(x) = (1, 0, 1, 1).
\]

Two observations are considered similar when their prediction vectors are
similar.

This is the main idea behind `CombinedClassifier`.

---

## How it works

The implementation follows five main stages.

### 1. Split the training data

By default, the training sample is divided into two parts:

\[
D_k
\quad\text{and}\quad
D_l.
\]

`D_k` is used to train the base classifiers.

`D_l` is used to construct and tune the aggregation rule.

With the default configuration:

```python
split_ratio=0.5
```

approximately half of the observations are assigned to each part.

This separation prevents the same observations from being used directly for
both fitting the base models and calibrating the aggregation rule.

---

### 2. Train the base classifiers

Each classifier is fitted using \(D_k\).

When `estimators=None`, the implementation uses:

```text
logistic_regression
decision_tree_classifier
svc
k_neighbors_classifier
```

For every observation in \(D_l\), all fitted classifiers make a prediction.

The predictions are stacked into a matrix:

\[
P_l =
\begin{bmatrix}
C_1(X_1) & C_2(X_1) & \cdots & C_M(X_1) \\
C_1(X_2) & C_2(X_2) & \cdots & C_M(X_2) \\
\vdots   & \vdots   &        & \vdots   \\
C_1(X_l) & C_2(X_l) & \cdots & C_M(X_l)
\end{bmatrix}.
\]

Each row is therefore an observation represented by the predictions of the
base classifiers.

---

### 3. Measure prediction disagreement

The next step measures the distance between prediction vectors.

The default distance is:

```python
distance="hamming"
```

For two prediction vectors \(p(x)\) and \(p(z)\), the normalized Hamming
distance is

\[
d_H(p(x), p(z))
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}
\left\{
C_m(x) \neq C_m(z)
\right\}.
\]

It measures the proportion of classifiers that disagree.

For example,

\[
p(x)=(1,0,1,1)
\]

and

\[
p(z)=(1,0,0,1)
\]

differ in one out of four positions, so

\[
d_H(p(x),p(z))=\frac{1}{4}=0.25.
\]

A distance of `0` means that all classifiers give the same predictions for
both observations.

---

### 4. Convert distance into similarity

Distances are transformed into weights using a kernel.

The default kernel is:

```python
kernel="rbf"
```

In the current implementation, the one-parameter kernel adapter first scales
the distance:

\[
D_h = hD,
\]

where \(h\) is the bandwidth parameter.

The default radial kernel then applies

\[
K(D_h)=\exp(-D_h).
\]

Therefore,

\[
w_i(x)
=
\exp
\left(
-h\,d(p(x),p(X_i))
\right).
\]

Observations whose prediction vectors are close to the query receive larger
weights.

Observations with very different classifier predictions receive smaller
weights.

---

## Relationship to the original combined classifier

The original combined-classifier procedure uses a strict consensus rule.

A training observation \(X_i\) participates in the vote for \(x\) when

\[
C_m(X_i)=C_m(x)
\]

for every classifier

\[
m=1,\ldots,M.
\]

The final class is determined by a majority vote among the matching
observations.

Conceptually:

```text
Query prediction vector
        │
        ▼
(1, 0, 1, 1)
        │
        ├── find training observations with
        │   the same prediction pattern
        │
        ▼
matching observations
        │
        ▼
majority class
```

`CombinedClassifier` generalizes this strict rule.

Instead of requiring exact agreement, it measures the amount of disagreement
and converts that distance into a continuous kernel weight:

```text
exact agreement
     │
     ▼
distance = 0
     │
     ▼
large weight
```

while

```text
many disagreements
     │
     ▼
larger distance
     │
     ▼
smaller weight
```

This produces a smoother form of consensual aggregation.

---

## 5. Weighted voting

For a query observation \(x\), let

\[
w_i(x)
\]

be the kernel similarity between its prediction vector and the prediction
vector of aggregation observation \(X_i\).

The score of class \(c\) is

\[
S_c(x)
=
\sum_{i=1}^{n_l}
w_i(x)
\mathbf{1}\{Y_i=c\}.
\]

The final prediction is

\[
\hat{Y}(x)
=
\operatorname*{arg\,max}_{c}
S_c(x).
\]

This is implemented by the default

```python
aggregator="weighted_vote"
```

strategy.

If every kernel weight is zero, the implementation falls back to the majority
class in the aggregation dataset.

---

## Bandwidth optimization

The bandwidth controls how quickly the influence of an observation decreases
as prediction-space disagreement increases.

A small or large value can substantially change the aggregation behavior, so
`CombinedClassifier` selects it automatically.

By default, the candidate values are

```python
np.linspace(0.001, 10.0, max_iter)
```

where

```python
max_iter=300
```

unless `bandwidth_list` is supplied explicitly.

For each candidate bandwidth, the aggregation dataset is evaluated using
K-fold cross-validation.

The default is

```python
n_cv=5
```

and the value minimizing the configured loss is stored as

```python
model.bandwidth_
```

The complete optimization history is available through

```python
model.optimization_outputs_
```

---

## Basic usage

```python
from kfc_procedure.cobra import CombinedClassifier

model = CombinedClassifier(
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The default pipeline is approximately:

```text
Training data
    │
    ├── 50% ──► Fit base classifiers
    │
    └── 50% ──► Aggregation dataset
                    │
                    ▼
              Base predictions
                    │
                    ▼
              Hamming distance
                    │
                    ▼
                 RBF kernel
                    │
                    ▼
            Optimize bandwidth
                    │
                    ▼
              Weighted vote
```

---

## Custom base classifiers

You can choose the classifiers used to construct prediction space.

```python
model = CombinedClassifier(
    estimators=[
        "logistic_regression",
        "random_forest_classifier",
        "svc",
        "k_neighbors_classifier",
    ],
    random_state=42,
)

model.fit(X_train, y_train)
```

Each estimator contributes one coordinate to prediction space.

If four estimators are used, each observation is represented by a
four-dimensional prediction vector.

---

## Estimator parameters

Parameters can be configured per estimator using `estimators_params`.

```python
model = CombinedClassifier(
    estimators=[
        "logistic_regression",
        "decision_tree_classifier",
        "k_neighbors_classifier",
    ],
    estimators_params={
        "logistic_regression": {
            "max_iter": 1000,
        },
        "decision_tree_classifier": {
            "max_depth": 6,
        },
        "k_neighbors_classifier": {
            "n_neighbors": 7,
        },
    },
    random_state=42,
)
```

---

## Available distance functions

The package currently registers several prediction-space distances.

| Name | Aliases | Description |
| --- | --- | --- |
| `hamming` | — | Proportion of classifier predictions that disagree |
| `euclidean` | `l2` | Euclidean distance |
| `manhattan` | `l1` | Manhattan distance |
| `minkowski` | `lp` | General Minkowski distance |
| `cosine` | — | Cosine distance |

For classifiers producing discrete labels, Hamming distance is the natural
default because it directly measures disagreement between classifiers.

```python
model = CombinedClassifier(
    distance="hamming",
)
```

---

## Available kernels

Several kernels can transform prediction-space distance into an aggregation
weight.

Available implementations include:

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
model = CombinedClassifier(
    kernel="epanechnikov",
)
```

or

```python
model = CombinedClassifier(
    kernel="triweight",
)
```

Different kernels determine how quickly observations lose influence as their
prediction vectors become less similar.

---

## Custom bandwidth grid

By default, bandwidth values are generated automatically.

You can provide your own search space:

```python
import numpy as np

model = CombinedClassifier(
    bandwidth_list=np.linspace(0.01, 5.0, 100),
    n_cv=5,
    random_state=42,
)
```

After fitting:

```python
print(model.bandwidth_)
```

returns the selected bandwidth.

You can inspect the optimization information with:

```python
model.optimization_outputs_
```

and the optimization history with:

```python
model.optimization_outputs_["history"]
```

---

## Using a separate aggregation dataset

You may provide the two datasets explicitly.

```python
model.fit(
    X_k,
    y_k,
    X_l=X_l,
    y_l=y_l,
)
```

Here:

- `X_k`, `y_k` train the base classifiers,
- `X_l`, `y_l` construct and optimize the combined classifier.

When these arguments are supplied, the automatic training split is skipped.

This is useful when you need complete control over the data partition.

---

## Using precomputed predictions

`CombinedClassifier` can also operate directly on a prediction matrix.

Set:

```python
as_predictions=True
```

when each column of `X` already contains the output of one base classifier.

```python
model = CombinedClassifier(
    distance="hamming",
    kernel="rbf",
    random_state=42,
)

model.fit(
    prediction_matrix,
    y,
    as_predictions=True,
)
```

For example,

```text
               Model 1   Model 2   Model 3
Sample 1           0         0         1
Sample 2           1         1         1
Sample 3           0         1         0
Sample 4           1         1         0
```

is already a valid prediction-space representation.

During inference, the query must be supplied in the same prediction-space
format.

```python
pred = model.predict(query_prediction_matrix)
```

This mode is useful when the base classifiers are trained outside the
`kfc-procedure` pipeline.

---

## Prediction probabilities

The estimator exposes the scikit-learn-style method:

```python
proba = model.predict_proba(X_test)
```

The returned array has shape

```text
(n_samples, n_classes)
```

and the class ordering is available from

```python
model.classes_
```

!!! note "Current implementation"
    Probability calculation is delegated to the selected aggregator.
    With the current default `weighted_vote` implementation, its
    single-query `aggregate_proba()` computes class frequencies rather than
    applying the supplied kernel weights. Therefore, treat `predict_proba()`
    as aggregator-dependent behavior rather than assuming it is a calibrated
    probability estimate.

---

## Important fitted attributes

After calling `fit()`, several useful attributes are available.

| Attribute | Description |
| --- | --- |
| `estimators_` | Fitted base classifiers |
| `pred_l_` | Prediction-space representation of the aggregation data |
| `distance_matrix_` | Pairwise distance matrix for aggregation observations |
| `bandwidth_` | Selected kernel bandwidth |
| `classes_` | Observed class labels |
| `global_majority_class_` | Fallback class when no positive weight is available |
| `cv_folds_` | Cross-validation folds used during tuning |
| `optimization_outputs_` | Optimizer score, bandwidth, and search history |

For example:

```python
model.fit(X_train, y_train)

print(model.bandwidth_)
print(model.classes_)
print(model.optimization_outputs_["history"])
```

---

## `CombinedClassifierFast`

The package also provides

```python
CombinedClassifierFast
```

which subclasses `CombinedClassifier` and can optionally use
[FAISS](https://github.com/facebookresearch/faiss) to restrict aggregation to
nearby prediction vectors.

```python
from kfc_procedure.cobra.combined_classifier import CombinedClassifierFast

model = CombinedClassifierFast(
    use_faiss=True,
    faiss_k=100,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Instead of comparing a query against every aggregation observation, FAISS
retrieves approximately the nearest prediction-space neighbors.

This can reduce prediction cost when the aggregation dataset is large.

!!! note
    FAISS acceleration is used only when `faiss` is installed and
    `use_faiss=True`. Otherwise the class falls back to the standard
    `CombinedClassifier` prediction path.

---

## Original consensus interpretation

The original combined classifier can be understood as defining a partition of
the input observations through the outputs of the base classifiers.

Suppose three classifiers produce

```text
C1(x) = 1
C2(x) = 0
C3(x) = 1
```

The query belongs to the prediction-space cell

```text
(1, 0, 1)
```

The original rule searches the training sample for observations with exactly
the same prediction pattern:

```text
                     Prediction vector     True class
Observation 1           (1, 0, 1)              0
Observation 2           (1, 0, 1)              1
Observation 3           (1, 1, 1)              1
Observation 4           (1, 0, 1)              0
Observation 5           (0, 0, 1)              1
```

Only observations 1, 2, and 4 belong to the same prediction cell.

Their labels are

```text
0, 1, 0
```

so the majority vote gives

```text
prediction = 0
```

The kernel implementation follows the same intuition but replaces the
hard same-cell condition with a continuous notion of similarity.

---

## Mathematical view

Let

\[
m(x)
=
\left(
m^{(1)}(x),
\ldots,
m^{(M)}(x)
\right)
\]

be the prediction vector generated by \(M\) base classifiers.

The strict consensus version assigns class \(1\) in binary classification
when

\[
\sum_{i=1}^{n}
\mathbf{1}\{m(X_i)=m(x)\}
\mathbf{1}\{Y_i=1\}
>
\sum_{i=1}^{n}
\mathbf{1}\{m(X_i)=m(x)\}
\mathbf{1}\{Y_i=0\}.
\]

In the kernelized implementation, the strict indicator is replaced by a
similarity weight.

For the default configuration,

\[
w_i(x)
=
\exp
\left(
-h\,
d_H(m(X_i),m(x))
\right).
\]

The class prediction becomes

\[
\hat y(x)
=
\operatorname*{arg\,max}_{c}
\sum_{i=1}^{n_l}
w_i(x)
\mathbf{1}\{Y_i=c\}.
\]

This lets observations with partial classifier agreement contribute to the
decision instead of requiring complete agreement.

---

## When is `CombinedClassifier` useful?

`CombinedClassifier` is especially useful when several classifiers capture
different aspects of the data and selecting one model in advance is difficult.

It can also be useful when you want the ensemble decision to depend on
**agreement patterns** between classifiers rather than a simple global
majority vote.

A standard voting classifier asks:

```text
What class do most models predict for x?
```

`CombinedClassifier` instead asks:

```text
Which previously observed samples produced a prediction pattern similar
to x, and what were their actual labels?
```

That distinction is central to consensual aggregation.

---

## CombinedClassifier vs majority voting

These two approaches should not be confused.

### Majority voting

For one query:

```text
Classifier A -> 1
Classifier B -> 1
Classifier C -> 0
```

Majority voting immediately predicts:

```text
1
```

### CombinedClassifier

The same predictions define

```text
(1, 1, 0)
```

as the query's location in prediction space.

The algorithm then compares this pattern with the prediction patterns of the
aggregation observations and combines their **true labels** using similarity
weights.

Therefore, the base classifiers define the neighborhood, while the observed
training labels determine the final decision.

---

## Summary

`CombinedClassifier` performs classification through consensual aggregation:

```text
Input features
     │
     ▼
Base classifiers
     │
     ▼
Prediction vector
     │
     ▼
Prediction-space distance
     │
     ▼
Kernel similarity
     │
     ▼
Weighted labels
     │
     ▼
Final class
```

The original method uses exact agreement between classifier predictions.
The implementation in `kfc-procedure` extends this idea with distances,
kernels, weighted voting, and automatically optimized bandwidths.

This makes the consensus mechanism configurable while preserving its central
idea: **classify a new observation using training observations whose ensemble
prediction behavior is similar to its own**.