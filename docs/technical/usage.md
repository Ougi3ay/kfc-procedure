# Usage

This page explains how to install, run, and use the project.

---

## 1. Installation

### Install from PyPI

```bash
pip install kfc-procedure
```

### Install with COBRA support

```bash
pip install "kfc-procedure[cobra]"
```

### Install for development

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
python -m pip install -e ".[dev]"
```

### Install all extras

```bash
python -m pip install -e ".[all]"
```

---

## 2. Importing

Install name:

```bash
pip install kfc-procedure
```

Python import name:

```python
import kfc_procedure
```

!!! note
    Python module names cannot contain hyphens, so the package uses `kfc_procedure` as the import name.

---

## 3. Regression with KFCRegressor

The advertised README example uses `combiner="mean"`, but a smoke test on the provided codebase showed that the current `MeanCombiner` may fail because `CStep` injects `random_state` into stateless combiners. Until that is fixed, `weighted_mean` is a safer choice.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from kfc_procedure import KFCRegressor

X, y = make_regression(
    n_samples=300,
    n_features=8,
    noise=0.2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="weighted_mean",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## 4. Classification with KFCClassifier

Use `stacking_classifier` as a safer current option than `majority_vote` because `majority_vote` has the same stateless-combiner issue described above.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from kfc_procedure import KFCClassifier

X, y = make_classification(
    n_samples=300,
    n_features=8,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="decision_tree_classifier",
    combiner="stacking_classifier",
    n_clusters=2,
    random_state=42,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

!!! warning "Classification probability prediction"
    `KFCClassifier.predict_proba` currently calls `FStep.predict_proba`, but the inspected `FStep` class does not implement this method. Use `predict` unless `FStep.predict_proba` is added.

---

## 5. Choosing divergences

Start with:

```python
divergences=["euclidean"]
```

Use additional divergences only when the data domain is valid.

| Divergence | Required input domain |
|---|---|
| `euclidean` | any real values |
| `gkl` | all values strictly greater than 0 |
| `is` | all values strictly greater than 0 |
| `logistic` | all values strictly between 0 and 1 |

Example with multiple divergences:

```python
model = KFCRegressor(
    divergences=["euclidean", "gkl"],
    local_model="random_forest_regressor",
    combiner="weighted_mean",
    n_clusters=3,
    random_state=42,
)
```

Before using `gkl` or `is`, ensure all feature values are positive.

---

## 6. Passing parameters

### Local model parameters

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={
        "n_estimators": 100,
        "max_depth": 5,
    },
    combiner="weighted_mean",
    n_clusters=3,
    random_state=42,
)
```

### Combiner parameters

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="weighted_mean",
    combiner_params={
        "fit_intercept": True,
    },
    n_clusters=3,
    random_state=42,
)
```

---

## 7. GradientCOBRA usage

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from kfc_procedure.cobra import GradientCOBRA

X, y = make_regression(
    n_samples=300,
    n_features=6,
    noise=0.3,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

model = GradientCOBRA(
    estimators=["linear_regression", "ridge", "random_forest_regressor"],
    kernel="rbf",
    distance="euclidean",
    loss="mse",
    opt_method="grid",
    max_iter=50,
    n_cv=5,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
```

---

## 8. MixCOBRA usage

```python
from kfc_procedure.cobra import MixCOBRARegressor

model = MixCOBRARegressor(
    estimators=["linear_regression", "ridge", "random_forest_regressor"],
    distance="euclidean",
    kernel="radial",
    loss="mse",
    opt_method="grid",
    max_iter=30,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 9. CombinedClassifier usage

```python
from kfc_procedure.cobra import CombinedClassifier

clf = CombinedClassifier(
    estimators=["logistic_regression", "decision_tree_classifier", "random_forest_classifier"],
    distance="hamming",
    kernel="cobra",
    loss="log_loss",
    opt_method="grid",
    max_iter=30,
    random_state=42,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
```

---

## 10. Running tests

Install development dependencies:

```bash
python -m pip install -e ".[dev,cobra]"
```

Run tests:

```bash
pytest
```

If needed, run with explicit source path:

```bash
PYTHONPATH=src pytest
```

---

## 11. Building documentation

Install documentation tools:

```bash
pip install mkdocs mkdocs-material pymdown-extensions "mkdocstrings[python]"
```

Run local documentation server:

```bash
mkdocs serve -f mkdocs.yml
```

Open:

```text
http://127.0.0.1:8000
```

---

## 12. Recommended first experiments

### Stable regression start

```python
KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="weighted_mean",
    n_clusters=2,
    random_state=42,
)
```

### Stable classification start

```python
KFCClassifier(
    divergences=["euclidean"],
    local_model="decision_tree_classifier",
    combiner="stacking_classifier",
    n_clusters=2,
    random_state=42,
)
```

### COBRA regression start

```python
GradientCOBRA(
    estimators=["linear_regression", "ridge", "random_forest_regressor"],
    kernel="rbf",
    distance="euclidean",
    loss="mse",
    opt_method="grid",
    max_iter=50,
    random_state=42,
)
```
