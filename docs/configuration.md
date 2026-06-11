# Configuration Guide


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


The package is configured primarily through constructor parameters and factory names. No required external configuration file or required environment variable was observed in the source code.

## KFCProcedure configuration

```python
KFCRegressor(
    divergences=["euclidean"],
    local_model="linear_regression",
    combiner="mean",
    n_clusters=3,
    random_state=42,
)
```

### Important parameters

| Parameter | Applies to | Recommendation |
|---|---|---|
| `divergences` | KFC | Start with `euclidean`; use `gkl`, `is`, or `logistic` only after scaling data into the required domain. |
| `local_model` | KFC | Use names from `LocalModelFactory.available()`. |
| `combiner` | KFC | Use task-compatible names from `CombinerFactory.available()`. |
| `n_clusters` | KFC | Tune based on sample size and local model stability. |
| `random_state` | KFC/COBRA | Set in experiments for reproducibility. |
| `max_iter` | KFC/COBRA | Reducing this can make notebooks faster during debugging. |
| `n_cv` | COBRA | Controls cross-validation folds during parameter optimization. |
| `bandwidth_list` | GradientCOBRA/CombinedClassifier | Use a narrowed list when grid search is slow. |

## Divergence domain configuration

| Divergence | Data preparation requirement |
|---|---|
| `euclidean` | Works with real-valued arrays. |
| `gkl` | Use strictly positive features. |
| `is` | Use strictly positive features. |
| `logistic` | Use values in `(0, 1)`. |

Example scaling for logistic divergence:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X01 = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6)).fit_transform(X)
```

## Runtime configuration

There are no package-specific environment variables in the inspected implementation. Runtime behavior is controlled by Python constructor arguments.

## Packaging configuration

The project uses `pyproject.toml` with a `src/` layout:

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["kfc_procedure*"]
```

## Documentation build configuration

This ZIP includes `mkdocs.yml` at the root. To preview the docs:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```
