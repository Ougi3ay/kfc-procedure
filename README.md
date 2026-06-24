# KFCProcedure

[![PyPI version](https://img.shields.io/pypi/v/kfc-procedure.svg)](https://pypi.org/project/kfc-procedure/)
[![Python versions](https://img.shields.io/pypi/pyversions/kfc-procedure.svg)](https://pypi.org/project/kfc-procedure/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://ougi3ay.github.io/kfc-procedure/)

`kfc-procedure` is a Python package for **clusterwise predictive modeling** and **COBRA-based ensemble aggregation**.

It implements a modular **K-step → F-step → C-step** learning pipeline:

* **K-step:** cluster the input data using one or more Bregman divergences.
* **F-step:** train local supervised models inside the induced clusters.
* **C-step:** combine local predictions using an aggregation strategy.

The package is designed for research, experimentation, and extensible machine learning workflows involving heterogeneous data, local modeling, and ensemble aggregation.

> PyPI package name: `kfc-procedure`
> Python import name: `kfc_procedure`
> Python version: `>=3.11`

---

## Installation

Install from PyPI:

```bash
pip install kfc-procedure
```

Install with optional COBRA dependencies:

```bash
pip install "kfc-procedure[cobra]"
```

Install for development:

```bash
pip install "kfc-procedure[dev]"
```

Install all optional dependencies:

```bash
pip install "kfc-procedure[all]"
```

---

## Main Features

* Scikit-learn-style estimators for regression and classification.
* Modular KFCProcedure pipeline:

  * divergence-based clustering,
  * cluster-local supervised learning,
  * prediction aggregation.
* Built-in Bregman divergences:

  * `euclidean`
  * `gkl`
  * `is`
  * `logistic`
* Registry-based factories for extensibility.
* COBRA-based estimators:

  * `GradientCOBRA`
  * `MixCOBRARegressor`
  * `CombinedClassifier`
  * `SuperLearner`
* Automatic registration of compatible scikit-learn estimators as local models.
* Extensible architecture for adding new divergences, local estimators, kernels, distances, losses, optimizers, and aggregation strategies.

---

## Research Context

This package was developed as the software implementation component of the engineering degree thesis:

**Python Libraries for Clusterwise Predictive Models: KFCProcedure and GradientCOBRA**

The thesis studies the design and implementation of a reusable Python framework for clusterwise supervised learning and COBRA-based aggregation. The package contains two connected subsystems:

### 1. KFCProcedure subsystem

The KFCProcedure subsystem follows a three-stage pipeline:

* **K-step:** applies divergence-based clustering to detect latent structures in the input space.
* **F-step:** trains local predictive models inside each cluster.
* **C-step:** combines local prediction outputs into a final prediction.

### 2. GradientCOBRA subsystem

The GradientCOBRA subsystem provides COBRA-style aggregation methods, including:

* `GradientCOBRA` for regression aggregation,
* `MixCOBRARegressor` for mixed input-space and prediction-space regression aggregation,
* `CombinedClassifier` for classification aggregation,
* `SuperLearner` for meta-learning-based ensemble prediction.

The package should be understood as a reusable and extensible research framework. Its performance depends on the dataset structure, selected divergences, local models, combiners, hyperparameters, and preprocessing.

---

## Quick Start

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
    local_model="ridge",
    combiner="mean",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
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

print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Regression with `GradientCOBRA`

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
    optimizer="grid",
    max_iter=50,
    n_cv=5,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
```

---

## Main API

### Top-level estimators

```python
from kfc_procedure import KFCProcedure, KFCRegressor, KFCClassifier
```

| Class           | Purpose                                                                   |
| --------------- | ------------------------------------------------------------------------- |
| `KFCProcedure`  | Base estimator implementing the full K-step, F-step, and C-step pipeline. |
| `KFCRegressor`  | Regression wrapper around `KFCProcedure`.                                 |
| `KFCClassifier` | Classification wrapper around `KFCProcedure`.                             |

### COBRA estimators

```python
from kfc_procedure.cobra import (
    GradientCOBRA,
    MixCOBRARegressor,
    CombinedClassifier,
    SuperLearner,
)
```

| Class                | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `GradientCOBRA`      | COBRA-style regressor with kernel weighting and bandwidth optimization. |
| `MixCOBRARegressor`  | Regressor combining input-space and prediction-space distances.         |
| `CombinedClassifier` | COBRA-style classification estimator.                                   |
| `SuperLearner`       | Super learner implementation using base learners and a meta-learner.    |

---

## Module Overview

| Layer                   | Main location                                           | Role                                                                                              |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Orchestration layer     | `kfc_procedure/kfc.py`                                  | Provides `KFCProcedure`, `KFCRegressor`, and `KFCClassifier`.                                     |
| KFC core layer          | `kfc_procedure/core/`                                   | Provides clustering, local learning, K-step, F-step, C-step, and combiners.                       |
| COBRA aggregation layer | `kfc_procedure/cobra/`                                  | Provides COBRA-based regression, classification, and super learner components.                    |
| Utility layer           | `kfc_procedure/utils/` and `kfc_procedure/cobra/utils/` | Provides helper functions for logging, component resolution, splitting, and estimator management. |

### KFC core modules

| Module             | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `core/clustering/` | Bregman K-Means clustering and divergence implementations.     |
| `core/ml/`         | Local model interface and scikit-learn estimator registration. |
| `core/steps/`      | K-step, F-step, and C-step execution logic.                    |
| `core/combiner/`   | Regression and classification aggregation strategies.          |
| `core/factory.py`  | Registry-based component construction.                         |

### COBRA modules

| Module                    | Purpose                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `cobra/core/distances/`   | Distance functions such as Euclidean, Manhattan, Minkowski, cosine, and Hamming.                                |
| `cobra/core/kernels/`     | Kernel functions such as radial, COBRA, Cauchy, exponential, triangular, biweight, triweight, and Epanechnikov. |
| `cobra/core/losses/`      | Loss functions such as MSE, MAE, Huber, quantile, hinge, and log loss.                                          |
| `cobra/core/optimizers/`  | Grid and gradient-based optimization components.                                                                |
| `cobra/core/splitters/`   | Holdout and split-overlap strategies.                                                                           |
| `cobra/core/aggregators/` | Weighted mean and weighted vote aggregation.                                                                    |
| `cobra/core/normalizers/` | Standard and min-max normalization.                                                                             |
| `cobra/core/cv/`          | K-fold, stratified K-fold, and time-series cross-validation.                                                    |

---

## Supported KFC Components

### Divergences

| Name        | Typical domain         | Notes                                    |
| ----------- | ---------------------- | ---------------------------------------- |
| `euclidean` | Real-valued data       | Squared Euclidean Bregman divergence.    |
| `gkl`       | Strictly positive data | Generalized Kullback-Leibler divergence. |
| `is`        | Strictly positive data | Itakura-Saito divergence.                |
| `logistic`  | Values in `(0, 1)`     | Logistic/Bernoulli Bregman divergence.   |

Use domain-specific divergences only when the input data satisfy the required domain constraints.

### Common local models

Local models are resolved through the `LocalModelFactory`. Compatible scikit-learn estimators are automatically registered using snake-case names.

Examples:

| Task                | Example local model names                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| Regression          | `linear_regression`, `ridge`, `lasso`, `random_forest_regressor`, `decision_tree_regressor`, `svr`             |
| Classification      | `logistic_regression`, `decision_tree_classifier`, `random_forest_classifier`, `svc`, `k_neighbors_classifier` |
| Baseline regression | `mean_regressor`, `dummy_mean`                                                                                 |

### Combiners

| Task           | Combiner names                                                             |
| -------------- | -------------------------------------------------------------------------- |
| Regression     | `mean`, `weighted_mean`, `stacking_regressor`, `gradientcobra`, `mixcobra` |
| Classification | `majority_vote`, `stacking_classifier`, `combined_classifier`              |

---

## Extending the Package

The package uses abstract interfaces and registry-based factories. New components can be added without changing the main KFC workflow.

### Extension points

| Pipeline stage  | Extension type                                                                    |
| --------------- | --------------------------------------------------------------------------------- |
| K-step          | Add a new Bregman divergence or clustering geometry.                              |
| F-step          | Add a new local estimator.                                                        |
| C-step          | Add a new combiner or aggregation rule.                                           |
| COBRA subsystem | Add a new kernel, distance, loss, optimizer, splitter, normalizer, or aggregator. |

### Example: add a new C-step combiner

```python
import numpy as np

from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory

@CombinerFactory.register("median", categories={"regression"})
class MedianCombiner(BaseCombiner):
    def fit(self, X, y=None):
        return self

    def combine(self, X):
        return np.median(X, axis=1)
```

After registration, the combiner can be selected by name:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="median",
    n_clusters=3,
    random_state=42,
)
```

This design allows researchers and developers to test new aggregation strategies inside the same KFCProcedure pipeline.

---

## Notes on `predict_proba`

`KFCClassifier.predict_proba(X)` is available only when the full classification path supports probability prediction:

* the selected local model must support `predict_proba`, and
* the selected C-step combiner must implement `predict_proba`.

For hard-label voting, use `majority_vote`. For probability-based outputs, choose local models and combiners that support probability prediction.

---

## Known Implementation Notes

* `KFCProcedure.fit` performs an internal train-test split.
* For classification, the internal split is stratified by the target labels.
* Local classification models may fail when a cluster contains only one class and the selected estimator requires at least two classes.
* Some divergences require restricted input domains:

  * `gkl` and `is` require positive inputs,
  * `logistic` requires values in `(0, 1)`.
* The package follows a scikit-learn-like interface, but not every estimator implements every optional scikit-learn method.
* Experimental performance depends on dataset structure, preprocessing, divergences, local models, combiners, and hyperparameters.

---

## Development

Clone the repository:

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
```

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Build the package:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The project uses `README.md` as the PyPI long description:

```toml
[project]
readme = "README.md"
```

---

## Citation

If you find `kfc-procedure` helpful, please consider citing the main methodological papers:

* S. Has (2023), [Gradient COBRA: A kernel-based consensual aggregation for regression](https://doi.org/10.52933/jdssv.v3i2.70).
* S. Has, A. Fischer, and M. Mougeot (2021), [KFC: A clusterwise supervised learning procedure based on the aggregation of distances](https://doi.org/10.1080/00949655.2021.1891539).
* G. Biau, A. Fischer, B. Guedj, and J. D. Malley (2016), [COBRA: A Combined Regression Strategy](https://doi.org/10.1016/j.jmva.2015.04.007).

You may also cite this software package as:

```text
Pov, O. (2026). KFCProcedure: Python package for clusterwise predictive modeling and COBRA-based ensemble aggregation (Version 0.1.1) [Computer software]. GitHub. https://github.com/Ougi3ay/kfc-procedure
```

```bibtex
@software{kfcprocedure2026,
  author  = {Pov, Ougi},
  title   = {KFCProcedure: Python Package for Clusterwise Predictive Modeling and COBRA-Based Ensemble Aggregation},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/Ougi3ay/kfc-procedure},
  license = {MIT},
  note    = {Python package}
}
```

Citation metadata is also provided in `CITATION.cff`.

---

## References

* United Nations General Assembly. (2015). *Transforming our world: The 2030 Agenda for Sustainable Development* (A/RES/70/1). https://sdgs.un.org/2030agenda

* Ali, S., Tirumala, S. S., & Sarrafzadeh, A. (2015). Ensemble learning methods for decision making: Status and future prospects. In *2015 International Conference on Machine Learning and Cybernetics (ICMLC)*, 1, 211--216. https://doi.org/10.1109/ICMLC.2015.7340924

* Biau, G., Fischer, A., Guedj, B., & Malley, J. D. (2016). COBRA: A Combined Regression Strategy. *Journal of Multivariate Analysis*, 146, 18--28. https://doi.org/10.1016/j.jmva.2015.04.007

* Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123--140. https://doi.org/10.1007/BF00058655

* Cagnini, H. E. L., das Dôres, S. C. N., Freitas, A. A., & Barros, R. C. (2023). A survey of evolutionary algorithms for supervised ensemble learning. *The Knowledge Engineering Review*, 38, e1, 1--43. https://doi.org/10.1017/S0269888923000024

* Fischer, A., & Mougeot, M. (2019). Aggregation using input--output trade-off. *Journal of Statistical Planning and Inference*, 200, 1--19. https://doi.org/10.1016/j.jspi.2018.08.001

* Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119--139. https://doi.org/10.1006/jcss.1997.1504

* Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1993). Design patterns: Abstraction and reuse of object-oriented design. In O. M. Nierstrasz (Ed.), *ECOOP '93 -- Object-Oriented Programming* (Lecture Notes in Computer Science, Vol. 707, pp. 406--431). Springer. https://doi.org/10.1007/3-540-47910-4_21

* Has, S., Fischer, A., & Mougeot, M. (2021). KFC: A clusterwise supervised learning procedure based on the aggregation of distances. *Journal of Statistical Computation and Simulation*, 91(11), 2307--2327. https://doi.org/10.1080/00949655.2021.1891539

* Has, S. (2023). Gradient COBRA: A kernel-based consensual aggregation for regression. *Journal of Data Science, Statistics, and Visualisation*, 3(2). https://doi.org/10.52933/jdssv.v3i2.70

* Mojirsheibani, M. (1999). Combining classifiers via discretization. *Journal of the American Statistical Association*, 94(446), 600--609. https://doi.org/10.1080/01621459.1999.10474154

* Oshaibi, F. M., AlKhanafseh, M., & Surakhi, O. (2024). Software effort estimation using ensemble learning. *Preprints.org*. https://doi.org/10.20944/preprints202403.0437.v1

* Wood, D., Mu, T., Webb, A. M., Reeve, H. W. J., Luján, M., & Brown, G. (2023). A unified theory of diversity in ensemble learning. *Journal of Machine Learning Research*, 24(359), 1--49. https://jmlr.org/papers/v24/23-0041.html

* Yang, Y., Lv, H., & Chen, N. (2023). A survey on ensemble learning under the era of deep learning. *Artificial Intelligence Review*, 56(6), 5545--5589. https://doi.org/10.1007/s10462-022-10283-5

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.

---

## Links

* Homepage: https://github.com/Ougi3ay/kfc-procedure
* PyPI: https://pypi.org/project/kfc-procedure/
* Documentation: https://ougi3ay.github.io/kfc-procedure/
* Issues: https://github.com/Ougi3ay/kfc-procedure/issues
