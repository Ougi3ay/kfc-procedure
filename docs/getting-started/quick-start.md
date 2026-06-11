# Quick Start


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


The examples below use names observed in the actual factory registries. In particular, use `linear_regression` rather than `linear`, and `logistic_regression` rather than `logistic`.

## KFC regression

```python
import numpy as np
from kfc_procedure import KFCRegressor

rng = np.random.RandomState(42)
X = rng.randn(120, 4)
y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=120)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    n_clusters=2,
    random_state=42,
)

model.fit(X, y)
y_pred = model.predict(X[:5])
print(y_pred)
```

Expected output: a one-dimensional NumPy array with five regression predictions.

## KFC classification

```python
import numpy as np
from kfc_procedure import KFCClassifier

rng = np.random.RandomState(42)
X = rng.randn(120, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="majority_vote",
    n_clusters=2,
    random_state=42,
)

clf.fit(X, y)
y_pred = clf.predict(X[:5])
print(y_pred)
```

Expected output: class labels for the five samples. The inspected high-level `KFCClassifier.predict_proba()` path is incomplete because `FStep.predict_proba()` is not implemented.

## GradientCOBRA regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from kfc_procedure.cobra import GradientCOBRA

X, y = make_regression(
    n_samples=200,
    n_features=8,
    noise=0.2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = GradientCOBRA(
    estimators=["linear_regression", "ridge"],
    kernel="rbf",
    distance="euclidean",
    optimizer="grid",
    max_iter=20,
    n_cv=3,
    n_jobs=1,
    random_state=42,
)

model.fit(X_train, y_train)
print(model.predict(X_test[:5]))
```

## CombinedClassifier

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from kfc_procedure.cobra import CombinedClassifier

X, y = make_classification(
    n_samples=200,
    n_features=8,
    n_informative=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

clf = CombinedClassifier(
    estimators=["logistic_regression", "decision_tree_classifier"],
    distance="hamming",
    kernel="rbf",
    aggregator="weighted_vote",
    optimizer="grid",
    max_iter=20,
    n_cv=3,
    n_jobs=1,
    random_state=42,
)

clf.fit(X_train, y_train)
print(clf.predict(X_test[:5]))
print(clf.predict_proba(X_test[:5]))
```

## Precomputed prediction features

COBRA estimators can be trained directly on prediction-space features by setting `as_predictions=True` in `fit`.

```python
import numpy as np
from kfc_procedure.cobra import GradientCOBRA

# Rows are samples; columns are outputs from base models.
P = np.column_stack([
    np.linspace(0.0, 1.0, 100),
    np.linspace(0.1, 1.1, 100),
])
y = P.mean(axis=1) + np.random.RandomState(0).normal(scale=0.01, size=100)

model = GradientCOBRA(max_iter=10, n_cv=2, n_jobs=1, random_state=0)
model.fit(P, y, as_predictions=True)
print(model.predict(P[:3]))
```
