# Quick Start

## Workflow

```mermaid
flowchart LR
    A["Prepare X, y"] --> B["Split train/test"]
    B --> C["Create estimator"]
    C --> D["fit"]
    D --> E["predict"]
    E --> F["evaluate"]
```

## Regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from kfc_procedure import KFCRegressor

X, y = make_regression(n_samples=300, n_features=8, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    n_clusters=3,
    random_state=42,
)
model.fit(X_train, y_train)
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
```

## Classification

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from kfc_procedure import KFCClassifier

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="decision_tree_classifier",
    combiner="majority_vote",
    n_clusters=2,
    random_state=42,
)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
```

## Main parameters

| Parameter | Meaning | Example |
|---|---|---|
| `divergences` | list of Bregman divergences | `["euclidean"]` |
| `local_model` | supervised model fitted per cluster | `"linear_regression"` |
| `combiner` | final aggregation strategy | `"mean"` |
| `n_clusters` | number of clusters per divergence | `3` |
| `random_state` | reproducibility seed | `42` |

!!! warning "Divergence domains"
    `gkl` and `is` require positive data. `logistic` requires values in `(0, 1)`.
