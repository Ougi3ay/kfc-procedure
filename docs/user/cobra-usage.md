# COBRA Usage

The package includes COBRA-style estimators under `kfc_procedure.cobra`.

| Estimator | Task | Summary |
|---|---|---|
| `GradientCOBRA` | Regression | prediction-space kernel aggregation with optimized bandwidth |
| `MixCOBRARegressor` | Regression | mixes input-space and prediction-space distances |
| `CombinedClassifier` | Classification | kernel-weighted voting in prediction space |
| `SuperLearner` | Regression/stacking | base learners plus meta learners |

## GradientCOBRA

```python
from kfc_procedure.cobra import GradientCOBRA

model = GradientCOBRA(kernel="rbf", distance="euclidean", max_iter=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## MixCOBRARegressor

```python
from kfc_procedure.cobra import MixCOBRARegressor

model = MixCOBRARegressor(distance="euclidean", kernel="rbf", max_iter=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## CombinedClassifier

```python
from kfc_procedure.cobra import CombinedClassifier

clf = CombinedClassifier(distance="hamming", kernel="rbf", max_iter=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```
