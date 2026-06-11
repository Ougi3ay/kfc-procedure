# Source Analysis Report


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Files analyzed

The documentation was produced from:

- `README.md`;
- `pyproject.toml`;
- `requirements.txt`;
- package source under `src/kfc_procedure/`;
- notebooks under `notebooks/`;
- uploaded experiment notebook `Experiment_kfc_procedure.ipynb`;
- tests under `tests/cobra/`;
- thesis files included with the provided project material.

## Public API extraction method

Public APIs were identified by importing modules under `kfc_procedure` and listing classes/functions whose names do not start with `_` and whose `__module__` matches the inspected module.

## Test result

```text
============================= 116 passed in 7.43s ==============================
```

## Important source/documentation mismatches

| Area | README or draft wording | Source-grounded wording |
|---|---|---|
| Local regression alias | `linear` | `linear_regression` |
| Local classification alias | `logistic` | `logistic_regression` |
| Combiner parameter | `aggregation` in README quick start | Constructor uses `combiner` |
| Stacking alias | `stacking` | `stacking_regressor` or `stacking_classifier` |
| Probability path | high-level KFC probability implied | `KFCClassifier.predict_proba()` path incomplete because `FStep.predict_proba()` is missing |

## Recommended next source changes

1. Add aliases if the README style should remain user-friendly.
2. Or update README examples to use exact factory names.
3. Implement `FStep.predict_proba()` for classification local models.
4. Add high-level KFC integration tests.
5. Add small example notebooks that can be run in CI.
