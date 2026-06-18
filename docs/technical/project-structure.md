# Project Structure

This page summarizes the repository layout and explains the role of important modules.

---

## Directory tree

```text
kfc-procedure/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   │   └── ci-cd.yml
│   ├── CODEOWNERS
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   └── kfc_procedure/
│       ├── __init__.py
│       ├── kfc.py
│       ├── core/
│       │   ├── factory.py
│       │   ├── clustering/
│       │   │   ├── bregman.py
│       │   │   └── divergences/
│       │   │       ├── base.py
│       │   │       ├── euclidean.py
│       │   │       ├── gkl.py
│       │   │       ├── itakura_saito.py
│       │   │       └── logistic.py
│       │   ├── steps/
│       │   │   ├── kstep.py
│       │   │   ├── fstep.py
│       │   │   └── cstep.py
│       │   ├── ml/
│       │   │   ├── base.py
│       │   │   └── sklearn.py
│       │   └── combiner/
│       │       ├── base.py
│       │       ├── regression/
│       │       └── classification/
│       ├── cobra/
│       │   ├── gradientcobra.py
│       │   ├── mixcobra.py
│       │   ├── combined_classifier.py
│       │   ├── superlearner.py
│       │   ├── core/
│       │   │   ├── adapters/
│       │   │   ├── aggregators/
│       │   │   ├── cv/
│       │   │   ├── distances/
│       │   │   ├── estimators/
│       │   │   ├── kernels/
│       │   │   ├── losses/
│       │   │   ├── normalizers/
│       │   │   ├── optimizers/
│       │   │   └── splitters/
│       │   └── utils/
│       └── utils/
├── tests/
│   └── cobra/
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── SECURITY.md
└── SUPPORT.md
```

---

## Root-level files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, build config, dependencies, optional extras, pytest configuration. |
| `requirements.txt` | Additional dependency list. |
| `README.md` | Main project description, installation, and examples. |
| `LICENSE` | MIT license. |
| `CITATION.cff` | Citation metadata for academic use. |
| `CONTRIBUTING.md` | Contribution guidelines. |
| `SECURITY.md` | Security policy. |
| `SUPPORT.md` | Support instructions. |

---

## Top-level package

### `src/kfc_procedure/__init__.py`

Exports the main public KFC estimators:

```python
from .kfc import KFCProcedure, KFCRegressor, KFCClassifier
```

### `src/kfc_procedure/kfc.py`

Implements:

- `KFCProcedure`
- `KFCRegressor`
- `KFCClassifier`

This is the main entry point for the K-step → F-step → C-step pipeline.

---

## Core modules

### `core/factory.py`

Defines `BaseFactory`, a generic registry-based factory.

Responsibilities:

- register classes under string aliases;
- create objects dynamically by name;
- filter constructor keyword arguments;
- track categories such as `regression`, `classification`, `search`, or `gradient`.

Important methods:

| Method | Purpose |
|---|---|
| `register(...)` | Decorator for adding classes to registry. |
| `create(name, **kwargs)` | Instantiate registered implementation. |
| `available()` | List registered aliases. |
| `contains(name)` | Check whether alias exists. |
| `supports(name, category)` | Check category compatibility. |
| `available_by_category(category)` | List aliases in category. |

---

## Clustering modules

### `core/clustering/bregman.py`

Implements `BregmanKMeans`.

Main responsibilities:

- validate divergence domain;
- initialize centroids;
- run Lloyd-style assignment/update iterations;
- compute distortion;
- expose `fit`, `predict`, `transform`, `fit_predict`, and `score`.

### `core/clustering/divergences/base.py`

Defines:

- `BaseBregmanDivergence`
- `BregmanDivergenceFactory`

The base class defines the common API:

```python
in_domain(X)
phi(X)
grad_phi(X)
distance(X, Y)
assign_clusters(X, centroids)
```

### Divergence implementations

| File | Registered name | Class |
|---|---|---|
| `euclidean.py` | `euclidean` | `SquaredEuclidean` |
| `gkl.py` | `gkl` | `GKLDivergence` |
| `itakura_saito.py` | `is` | `ItakuraSaito` |
| `logistic.py` | `logistic` | `LogisticLoss` |

---

## Pipeline step modules

### `core/steps/kstep.py`

Defines `KStep`, the multi-divergence clustering stage.

Input:

- feature matrix `X`;
- list of divergence identifiers or objects.

Output:

- fitted `BregmanKMeans` models;
- cluster assignment dictionary.

### `core/steps/fstep.py`

Defines `FStep`, the cluster-local model fitting stage.

Responsibilities:

- train one local model for each divergence and cluster;
- use `LocalModelFactory` to resolve string model names;
- produce the F-step prediction matrix.

### `core/steps/cstep.py`

Defines `CStep`, the final aggregation stage.

Responsibilities:

- resolve combiner with `CombinerFactory`;
- fit the combiner on the prediction matrix;
- predict final outputs.

---

## Local model modules

### `core/ml/base.py`

Defines:

- `BaseLocalModel`
- `LocalModelFactory`

### `core/ml/sklearn.py`

Defines:

- `MeanRegressor`
- `SklearnLocalModel`
- `register_all_sklearn_models()`

This module auto-registers compatible scikit-learn classifiers and regressors into the local model factory.

Examples of generated model names:

```text
linear_regression
ridge
lasso
random_forest_regressor
decision_tree_classifier
logistic_regression
```

---

## Combiner modules

### `core/combiner/base.py`

Defines:

- `BaseCombiner`
- `CombinerFactory`

### Regression combiners

| File | Registered name | Description |
|---|---|---|
| `regression/mean.py` | `mean` | Row-wise mean of predictions. |
| `regression/weighted_mean.py` | `weighted_mean` | Linear regression weighted average. |
| `regression/stacking.py` | `stacking_regressor` | Meta-regressor over prediction matrix. |
| `regression/gradientcobra.py` | `gradientcobra` | Wrapper around `GradientCOBRA`. |
| `regression/mixcobra.py` | `mixcobra` | Wrapper around `MixCOBRARegressor`. |

### Classification combiners

| File | Registered name | Description |
|---|---|---|
| `classification/majority_vote.py` | `majority_vote` | Hard voting over predicted labels. |
| `classification/stacking.py` | `stacking_classifier` | Logistic regression meta-classifier. |
| `classification/combined_classifier.py` | `combined_classifier` | Wrapper around `CombinedClassifier`. |

---

## COBRA modules

### Public COBRA estimators

| File | Class | Purpose |
|---|---|---|
| `cobra/gradientcobra.py` | `GradientCOBRA` | Kernel-weighted regression in prediction space. |
| `cobra/mixcobra.py` | `MixCOBRARegressor` | Regression using mixed input and prediction distances. |
| `cobra/combined_classifier.py` | `CombinedClassifier` | Kernel-weighted classification in prediction space. |
| `cobra/superlearner.py` | `SuperLearner` | Stacking/super learner regression implementation. |

### COBRA core component packages

| Package | Purpose |
|---|---|
| `adapters/` | Transform distance matrices with bandwidth or mixing parameters. |
| `aggregators/` | Weighted mean and weighted vote aggregation. |
| `cv/` | K-fold, stratified K-fold, and time-series CV wrappers. |
| `distances/` | Euclidean, Manhattan, Minkowski, Cosine, Hamming distances. |
| `estimators/` | Base estimator wrappers and scikit-learn estimator registration. |
| `kernels/` | Radial, exponential, cauchy, triangular, COBRA, and related kernels. |
| `losses/` | MSE, MAE, Huber, quantile, log loss, hinge loss. |
| `normalizers/` | Standard and min-max normalization components. |
| `optimizers/` | Grid search and gradient-based optimizers. |
| `splitters/` | Holdout and overlap data splitters. |

---

## Tests

The `tests/cobra/` directory contains unit tests for COBRA core components:

| Test file | Focus |
|---|---|
| `test_p01_splitters.py` | Splitter behavior. |
| `test_p02_estimators.py` | Estimator wrappers. |
| `test_p03_normalizers.py` | Normalizers. |
| `test_p04_distances.py` | Distance functions. |
| `test_p05_kernel_adapters.py` | Kernel adapters. |
| `test_p06_kernels.py` | Kernel functions. |
| `test_p07_losses.py` | Loss functions. |
| `test_p08_cv.py` | Cross-validation wrappers. |
| `test_p09_optimizers.py` | Optimizers. |
| `test_p10_aggregators.py` | Aggregators. |
