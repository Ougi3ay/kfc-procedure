# Troubleshooting


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'kfc_procedure'` | Package not installed or `src/` not on path | Run `pip install -e .` or set `PYTHONPATH=src`. |
| `Invalid local model` | Name not registered | Print `LocalModelFactory.available()` and use exact names. |
| `Invalid combiner` | Name not registered | Print `CombinerFactory.available()`. |
| `[GKL] Input outside valid domain` | Non-positive values with `gkl` | Use `euclidean` or scale features to positive values. |
| `[IS] Input outside valid domain` | Non-positive values with Itakura-Saito | Use positive-valued features only. |
| Logistic divergence domain error | Values not in `(0, 1)` | Apply min-max scaling with small epsilon bounds. |
| `predict_proba only available for classification` | Called probability prediction on regression model | Use `predict`. |
| `AttributeError: 'FStep' object has no attribute 'predict_proba'` | Incomplete high-level KFC probability path | Use `predict` or direct `CombinedClassifier.predict_proba()`. |
| COBRA fitting is slow | Large calibration set or grid | Reduce `max_iter`, reduce `n_cv`, or provide a smaller parameter list. |
| All weights become zero | Kernel/bandwidth too restrictive | Increase bandwidth range or use a different kernel. |
| README example fails with `linear` | Alias mismatch | Use `linear_regression`. |
| README example fails with `aggregation` | Parameter name mismatch | Use `combiner` in the current `KFCProcedure` constructor. |

## Debugging checklist

1. Confirm package import:

   ```python
   import kfc_procedure
   print(kfc_procedure.__file__)
   ```

2. Confirm factory names:

   ```python
   from kfc_procedure.core.ml.base import LocalModelFactory
   print(LocalModelFactory.available())
   ```

3. Check data domains before using non-Euclidean divergences.
4. Reduce data size and `max_iter` to isolate algorithmic errors.
5. Run the COBRA test suite to check core component health.
