# Quick Start

KFC Procedure provides estimators for classification, regression, and ensemble learning.

This page introduces the main public estimators:

- `CombinedClassifier`
- `MixCOBRARegressor`
- `GradientCOBRA`
- `KFCProcedure`
- `KFCClassifier`
- `KFCRegressor`

---

## CombinedClassifier

Use `CombinedClassifier` to combine predictions from multiple classification models.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from kfc_procedure.cobra import CombinedClassifier

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = CombinedClassifier(
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])
```

---

## MixCOBRARegressor

`MixCOBRARegressor` combines information from the feature space and prediction
space to produce regression predictions.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from kfc_procedure.cobra import MixCOBRARegressor

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = MixCOBRARegressor(
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])
```

---

## GradientCOBRA

`GradientCOBRA` is a regression ensemble that combines predictions using
distance, kernel, and aggregation components.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from kfc_procedure.cobra import GradientCOBRA

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = GradientCOBRA(
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])
```

---

## KFCProcedure

`KFCProcedure` is the general three-stage KFC estimator:

```text
K-step → F-step → C-step
```

You explicitly define whether the task is regression or classification.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from kfc_procedure import KFCProcedure

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = KFCProcedure(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    task="regression",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])
```

---

## KFCClassifier

`KFCClassifier` is the classification-specific KFC estimator.

The task is automatically configured as classification.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from kfc_procedure import KFCClassifier

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="majority_vote",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])
```

---

## KFCRegressor

`KFCRegressor` is the regression-specific KFC estimator.

The task is automatically configured as regression.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from kfc_procedure import KFCRegressor

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])
```

---

## Which estimator should I use?

| Estimator | Task | Purpose |
|---|---|---|
| `CombinedClassifier` | Classification | Combine multiple classifiers |
| `MixCOBRARegressor` | Regression | Mix feature-space and prediction-space information |
| `GradientCOBRA` | Regression | Kernel-based COBRA ensemble |
| `KFCProcedure` | Both | General configurable KFC pipeline |
| `KFCClassifier` | Classification | KFC classification pipeline |
| `KFCRegressor` | Regression | KFC regression pipeline |