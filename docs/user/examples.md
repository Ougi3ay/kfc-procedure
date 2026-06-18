# Examples

## Random forest local models

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={"n_estimators": 100, "max_depth": 5},
    combiner="weighted_mean",
    n_clusters=3,
    random_state=42,
)
```

## Multi-divergence regression

```python
model = KFCRegressor(
    divergences=["euclidean", "gkl"],
    local_model="linear_regression",
    combiner="stacking_regressor",
    n_clusters=3,
    random_state=42,
)
```

## Classification with stacking

```python
from kfc_procedure import KFCClassifier

clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="random_forest_classifier",
    combiner="stacking_classifier",
    n_clusters=2,
    random_state=42,
)
```

## Save and load

```python
import joblib
joblib.dump(model, "kfc_model.joblib")
loaded = joblib.load("kfc_model.joblib")
```
