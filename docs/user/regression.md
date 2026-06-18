# Regression

Use `KFCRegressor` for continuous target variables.

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    n_clusters=3,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Regression combiners

| Combiner | Behavior |
|---|---|
| `mean` | arithmetic mean across prediction columns |
| `weighted_mean` | OLS learns weights over prediction columns |
| `stacking_regressor` | meta-regressor over prediction matrix |
| `gradientcobra` | COBRA kernel-weighted aggregation |
| `mixcobra` | COBRA aggregation using input and prediction spaces |

## Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```
