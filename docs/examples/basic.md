# Examples


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Basic example: KFC regression

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from kfc_procedure import KFCRegressor

rng = np.random.RandomState(0)
X = rng.randn(200, 6)
y = 1.5 * X[:, 0] - 2.0 * X[:, 2] + rng.normal(scale=0.2, size=200)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    n_clusters=3,
    random_state=0,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(mean_squared_error(y_test, pred))
```

## Intermediate example: task-compatible component selection

```python
from kfc_procedure.core.ml.base import LocalModelFactory
from kfc_procedure.core.combiner.base import CombinerFactory

print(LocalModelFactory.available())
print(CombinerFactory.available())
```

For classification, combine classifier-compatible models and combiners:

```python
from kfc_procedure import KFCClassifier

clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="random_forest_classifier",
    local_model_params={"n_estimators": 50},
    combiner="majority_vote",
    n_clusters=3,
    random_state=0,
)
```

## Advanced example: GradientCOBRA with custom grid

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from kfc_procedure.cobra import GradientCOBRA

X, y = make_regression(n_samples=300, n_features=10, noise=0.5, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = GradientCOBRA(
    estimators=["linear_regression", "ridge", "random_forest_regressor"],
    estimators_params={
        "ridge": {"alpha": 1.0},
        "random_forest_regressor": {"n_estimators": 50, "random_state": 1},
    },
    distance="euclidean",
    kernel="rbf",
    optimizer="grid",
    bandwidth_list=np.linspace(0.01, 2.0, 25),
    n_cv=3,
    n_jobs=1,
    random_state=1,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## Real-world-style example: safe divergence selection

For heterogeneous tabular data, start with Euclidean divergence unless the data are explicitly transformed for positive-domain divergences.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from kfc_procedure import KFCRegressor

numeric_features = ["age", "income", "score"]
categorical_features = ["region", "segment"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

X_processed = preprocessor.fit_transform(df[numeric_features + categorical_features])
y = df["target"].to_numpy()

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="weighted_mean",
    n_clusters=4,
    random_state=42,
)
model.fit(X_processed, y)
```

This example uses a typical scikit-learn preprocessing pipeline before passing arrays to `kfc_procedure`.
