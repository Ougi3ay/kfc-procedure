# KFCProcedure

`kfc-procedure` is a Python package for clusterwise predictive modeling and COBRA-based ensemble aggregation. It implements a modular **K-step → F-step → C-step** learning pipeline, where data are clustered under one or more Bregman divergences, local predictive models are trained inside the induced clusters, and the resulting predictions are combined by an aggregation strategy.

The package is intended for research and development workflows involving heterogeneous data, local modeling, and ensemble aggregation.

> Package name on PyPI: `kfc-procedure`  
> Python import name: `kfc_procedure`

---

## Features

- Scikit-learn-style estimators for regression and classification.
- Three-stage KFCProcedure pipeline:
  - **K-step**: divergence-aware Bregman K-Means clustering.
  - **F-step**: cluster-local supervised learning.
  - **C-step**: prediction aggregation and consensus modeling.
- Built-in Bregman divergences:
  - `euclidean`
  - `gkl`
  - `is`
  - `logistic`
- Registry-based factories for selecting divergences, local models, combiners, kernels, distances, losses, optimizers, splitters, and aggregators.
- Modular COBRA components, including:
  - `GradientCOBRA`
  - `MixCOBRARegressor`
  - `CombinedClassifier`
  - `SuperLearner`
- Automatic registration of compatible scikit-learn estimators as local models.
- Extensible architecture for adding new divergences, estimators, and aggregation strategies.

---

## Installation

### Install from PyPI

```bash
pip install kfc-procedure
```

### Optional extras

```bash
pip install "kfc-procedure[cobra]"
pip install "kfc-procedure[dev]"
pip install "kfc-procedure[all]"
```

### Development installation

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
python -m pip install -e ".[dev]"
```

---

## Requirements

The package metadata declares the following runtime requirements:

- Python `>=3.11`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `xgboost`

Optional COBRA and development extras may install additional packages such as `numba`, `faiss-cpu`, `plotly`, `pytest`, `build`, `twine`, and `jupyter`.

---

## Quick start

### Regression with `KFCRegressor`

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
    combiner="mean",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))
```

### Classification with `KFCClassifier`

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
    combiner="majority_vote",
    n_clusters=2,
    random_state=42,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
```

### GradientCOBRA regression

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

print(mean_absolute_error(y_test, y_pred))
```

---

## Main API

### Top-level estimators

```python
from kfc_procedure import KFCProcedure, KFCRegressor, KFCClassifier
```

| Class | Purpose |
|---|---|
| `KFCProcedure` | Base estimator implementing the full K-step, F-step, and C-step pipeline. |
| `KFCRegressor` | Regression wrapper around `KFCProcedure`. |
| `KFCClassifier` | Classification wrapper around `KFCProcedure`. |

### COBRA estimators

```python
from kfc_procedure.cobra import (
    GradientCOBRA,
    MixCOBRARegressor,
    CombinedClassifier,
    SuperLearner,
)
```

| Class | Purpose |
|---|---|
| `GradientCOBRA` | COBRA-style regressor with kernel weighting and bandwidth optimization. |
| `MixCOBRARegressor` | Regression estimator combining input-space and prediction-space distances. |
| `CombinedClassifier` | COBRA-style classification estimator. |
| `SuperLearner` | Super learner implementation using base learners and a meta-learner. |

---

## KFCProcedure workflow

A fitted `KFCRegressor` or `KFCClassifier` follows this sequence:

1. **Input conversion**  
   Input arrays are converted to NumPy arrays.

2. **Internal split**  
   The data are split internally into two parts. One part is used for clustering and local model fitting; the other part is used for training the final combiner.

3. **K-step**  
   One `BregmanKMeans` model is fitted for each requested divergence.

4. **F-step**  
   A local supervised model is trained for each cluster produced by each divergence.

5. **C-step**  
   The divergence-level local predictions are combined into the final prediction.

6. **Prediction**  
   New samples are assigned to divergence-specific clusters, passed to the corresponding local models, and aggregated by the fitted combiner.

---

## Supported KFC components

### Divergences

| Name | Typical domain | Notes |
|---|---|---|
| `euclidean` | Real-valued data | Squared Euclidean Bregman divergence. |
| `gkl` | Strictly positive data | Generalized Kullback-Leibler divergence. |
| `is` | Strictly positive data | Itakura-Saito divergence. |
| `logistic` | Values in `(0, 1)` | Logistic/Bernoulli Bregman divergence. |

Use domain-specific divergences only when the input data satisfy the required domain constraints.

### Local models

Local models are resolved through the `LocalModelFactory`. The package auto-registers compatible scikit-learn estimators using snake-case names.

Common examples:

| Task | Example local model names |
|---|---|
| Regression | `linear_regression`, `ridge`, `lasso`, `random_forest_regressor`, `decision_tree_regressor`, `svr` |
| Classification | `logistic_regression`, `decision_tree_classifier`, `random_forest_classifier`, `svc`, `k_neighbors_classifier` |
| Baseline regression | `mean_regressor`, `dummy_mean` |

### Combiners

| Task | Combiner names |
|---|---|
| Regression | `mean`, `weighted_mean`, `stacking_regressor`, `gradientcobra`, `mixcobra` |
| Classification | `majority_vote`, `stacking_classifier`, `combined_classifier` |

---

## Notes on `predict_proba`

`KFCClassifier.predict_proba(X)` is available only when the full classification path supports probability prediction:

- the selected local model must support `predict_proba`, and
- the selected C-step combiner must implement `predict_proba`.

For example, `majority_vote` is intended for hard-label voting and should not be used when probability output is required.

---

## Project structure

```text
kfc-procedure/
├── pyproject.toml
├── README.md
├── requirements.txt
├── src/
│   └── kfc_procedure/
│       ├── kfc.py
│       ├── core/
│       │   ├── clustering/
│       │   ├── combiner/
│       │   ├── ml/
│       │   └── steps/
│       ├── cobra/
│       │   ├── core/
│       │   ├── gradientcobra.py
│       │   ├── mixcobra.py
│       │   ├── combined_classifier.py
│       │   └── superlearner.py
│       └── utils/
└── tests/
    └── cobra/
```

---

## Development

### Run tests

```bash
python -m pip install -e ".[dev]"
pytest
```

If running directly from a source checkout without installing the package, set `PYTHONPATH`:

```bash
PYTHONPATH=src pytest
```

### Build package

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The project uses `README.md` as the package long description through `pyproject.toml`:

```toml
[project]
readme = "README.md"
```

---

## Extending the package

The package uses registry-based factories. New components can be added by subclassing the relevant base class and registering the implementation under a string name.

Example pattern:

```python
from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory

@CombinerFactory.register("my_combiner", categories={"regression"})
class MyCombiner(BaseCombiner):
    def fit(self, X, y=None):
        return self

    def combine(self, X):
        return X.mean(axis=1)
```

Registered components can then be selected by name in the pipeline.

---

## Known implementation notes

- `KFCProcedure.fit` performs an internal `train_test_split`. For classification, the split is stratified by the target labels.
- Local classification models may fail when a cluster contains only one class and the chosen estimator requires at least two classes. Tree-based classifiers are usually safer for small clusters.
- Some divergence implementations require restricted input domains. For example, `gkl` and `is` require positive inputs, while `logistic` requires values in `(0, 1)`.
- The package follows a scikit-learn-like interface, but not every estimator fully implements every optional scikit-learn method.

---

## Citation and research context

The package is motivated by clusterwise supervised learning, Bregman divergence clustering, and COBRA-style ensemble aggregation. It is suitable for experiments where a single global model may not capture heterogeneous data structure.

Relevant methodological background includes:

- Bregman divergence clustering.
- Clusterwise predictive modeling.
- Consensus aggregation.
- COBRA and kernel-weighted aggregation.
- Super learner and stacking-based ensemble methods.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.

---

## Links

- Homepage: <https://github.com/Ougi3ay/kfc-procedure>
- Documentation: see the `docs/` directory in the repository.
