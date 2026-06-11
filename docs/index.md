# KFCProcedure Developer Documentation


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


`kfc_procedure` is a Python package for clusterwise supervised learning and COBRA-style ensemble aggregation. It contains two connected subsystems:

1. **KFCProcedure**: a three-stage clusterwise learning pipeline made of K-step clustering, F-step local model fitting, and C-step prediction aggregation.
2. **COBRA components**: reusable modules for distances, kernels, kernel adapters, losses, splitters, cross-validation, optimization, estimators, normalizers, and aggregation.

The package follows a modular, factory-driven design. Developers select algorithms with registry names such as `"euclidean"`, `"linear_regression"`, `"weighted_mean"`, `"rbf"`, or `"grid"`, and the corresponding component is created by a factory class.

## Project status observed

| Area | Status from source analysis |
|---|---|
| Package version | `0.1.0` in `pyproject.toml` |
| Python requirement | `>=3.11` |
| Package layout | `src/kfc_procedure/` |
| Main estimators | `KFCProcedure`, `KFCRegressor`, `KFCClassifier`, `GradientCOBRA`, `MixCOBRARegressor`, `CombinedClassifier`, `SuperLearner` |
| Automated tests observed | COBRA core tests only |
| Test result | `============================= 116 passed in 7.43s ==============================` |
| Known incomplete path | `KFCClassifier.predict_proba()` calls `FStep.predict_proba()`, which is not implemented in the inspected source |

## Documentation map

- [Installation](getting-started/installation.md): environment, editable install, verification, and test commands.
- [Quick Start](getting-started/quick-start.md): minimal working examples for KFC, GradientCOBRA, and CombinedClassifier.
- [Architecture Overview](architecture/overview.md): package structure, layers, and component dependencies.
- [Functional Workflow](architecture/workflow.md): KFC and COBRA execution flows.
- [API Reference](api/index.md): public classes, methods, functions, and factory registries.
- [Examples](examples/basic.md): basic, intermediate, advanced, and real-world-style examples.
- [Configuration Guide](configuration.md): runtime parameters and factory names.
- [Developer Guide](developer-guide/index.md): extending the package, testing, build process, and coding practices.
- [Mathematical Foundations](mathematical-foundations.md): Bregman divergences and COBRA aggregation notation.
- [Troubleshooting](troubleshooting.md): common runtime issues and fixes.
- [Full Developer Documentation](full-developer-documentation.md): single-file reference version.

## Target users

This documentation is written for package maintainers, thesis/research developers, contributors, and users who need to understand how to configure and extend the library safely.
