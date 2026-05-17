# KFC-Model: A Python Implementation of the KFC Procedure

KFC-Model is a modular Python library for clusterwise predictive modeling using the KFC procedure (K-step, F-step, C-step). It combines multiple clustering divergences, local models, and aggregation strategies for regression and classification tasks.

## Features

- KFC meta-estimator for clusterwise learning
- Modular `KStep`, `FStep`, and `CStep` components
- Support for Bregman K-Means divergences
- Local model factories for regression and classification
- Aggregation strategies including mean, stacking, and GradientCOBRA
- Easy extension with custom components

## Installation

Requirements:

- Python 3.11+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install the package locally:

```bash
python3 -m pip install -e .
```

## Quick Start

```python
import numpy as np
from kfc_procedure.kfc import KFCRegressor, KFCClassifier

# Example data
X = np.random.randn(200, 5)
y_reg = X[:, 0] * 2 + np.random.randn(200) * 0.1
y_clf = (y_reg > 0).astype(int)

# Regression example
model = KFCRegressor(
    divergences=["euclidean", "kl"],
    local_model="linear",
    aggregation="mean",
    random_state=42,
)
model.fit(X, y_reg)
y_pred = model.predict(X)

# Classification example
clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic",
    aggregation="majority_vote",
    random_state=42,
)
clf.fit(X, y_clf)
y_pred_clf = clf.predict(X)
proba = clf.predict_proba(X)
```

## Core Components

- `KStep`: fits clustering models using one or more Bregman divergences
- `FStep`: trains local models for each cluster and divergence
- `CStep`: aggregates local predictions into final outputs
- `KFCRegressor` / `KFCClassifier`: full meta-estimators exposing `fit`, `predict`, and `predict_proba`

## Configuration

### `divergences`

The divergences parameter accepts:

- a list of divergence names, e.g. `['euclidean', 'kl']`
- a list of config dictionaries, e.g. `[{ 'name': 'euclidean', 'n_clusters': 4 }]`

Available divergences: `'euclidean'`, `'kl'`, `'gkl'`, `'is'`, `'logistic'`

### `local_model`

The local_model parameter accepts:

- a model name string, e.g. `'linear'`, `'ridge'`

Supported regression models include: `linear`, `ridge`, `lasso`, `decision_tree`, `random_forest`.
Supported classification models include: `logistic`, `decision_tree`, `random_forest`.

### `aggregation`

The aggregation parameter accepts:

- an aggregation strategy name string, e.g. `'mean'`, `'stacking'`

Supported aggregators:

- Regression: `mean`, `weighted_mean`, `stacking`
- Classification: `majority_vote`, `stacking`, `combine_classifier`

## Project Structure

- `src/kfc_procedure/`: main package code
- `src/kfc_procedure/core/`: factories, clustering, ML wrappers, and aggregation strategies
- `src/kfc_procedure/steps/`: KFC step implementations
- `src/kfc_procedure/utils/`: resolution and validation helpers

## Contributing

Contributions, bug reports, and improvements are welcome. Use `pytest` for testing and follow the existing package layout for new components.

## License

MIT License

