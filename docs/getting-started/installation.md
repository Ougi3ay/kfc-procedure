# Installation Guide


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Requirements

The project metadata defines the following requirements:

| Requirement | Observed value |
|---|---|
| Python | `>=3.11` |
| Build backend | `setuptools.build_meta` |
| Runtime dependencies | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost` |
| Optional COBRA extras | `numba`, `faiss-cpu`, `plotly`, `xgboost`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` |
| Development extras | `pytest`, `build`, `twine`, `jupyter` |

The local `requirements.txt` additionally lists `gradientcobra`, `numba`, `faiss-cpu`, `pytest`, and `jupyter`.

## Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows PowerShell
python -m pip install --upgrade pip
```

## Install from the local repository

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
python -m pip install -e .
```

## Install optional dependency groups

```bash
python -m pip install -e ".[cobra]"
python -m pip install -e ".[dev]"
python -m pip install -e ".[all]"
```

## Install with `requirements.txt`

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use this form when reproducing the provided notebook environment.

## Verify installation

```python
from kfc_procedure import KFCProcedure, KFCRegressor, KFCClassifier
from kfc_procedure.core.clustering.divergences.base import BregmanDivergenceFactory
from kfc_procedure.core.ml.base import LocalModelFactory

print(BregmanDivergenceFactory.available())
print("linear_regression" in LocalModelFactory.available())
```

Expected behavior:

- the import should succeed;
- the divergence registry should include `euclidean`, `gkl`, `is`, and `logistic`;
- the local model registry should include scikit-learn-derived names such as `linear_regression`, `ridge`, and `logistic_regression`.

## Run tests

```bash
python -m pytest
```

When running without editable installation, add the source directory to `PYTHONPATH`:

```bash
PYTHONPATH=src python -m pytest tests/cobra
```

Observed result during documentation generation:

```text
============================= 116 passed in 7.43s ==============================
```

## Common installation issue

If `pytest` or a notebook raises `ModuleNotFoundError: No module named 'kfc_procedure'`, the package is not installed or `src/` is not on the Python path. Use editable installation or run with `PYTHONPATH=src`.
