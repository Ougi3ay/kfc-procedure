# Contributing to KFCProcedure

Thank you for your interest in contributing to `kfc-procedure`.

This project is a Python package for clusterwise predictive modeling and COBRA-based ensemble aggregation. Contributions should keep the package reliable, reproducible, and consistent with the scikit-learn style used throughout the public API.

## Ways to contribute

You can help by:

- reporting bugs,
- improving documentation,
- adding examples,
- improving tests,
- fixing compatibility issues,
- adding new divergences, local models, combiners, kernels, losses, distances, or optimizers.

Before starting a large change, open an issue first so the design can be discussed.

## Development setup

Clone the repository:

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,cobra]"
```

If optional dependencies are not needed, install only the development extras:

```bash
python -m pip install -e ".[dev]"
```

## Running tests

Run the full test suite:

```bash
pytest
```

If you are running tests directly from a source checkout without installing the package, set `PYTHONPATH`:

```bash
PYTHONPATH=src pytest
```

Pull requests should include tests for new behavior and should not break existing tests.

## Documentation

The documentation is built with MkDocs.

Preview locally:

```bash
mkdocs serve
```

Build locally:

```bash
mkdocs build
```

The public documentation site is:

```text
https://ougi3ay.github.io/kfc-procedure/
```

Documentation changes should be made in the `docs/` directory when available. The `README.md` should remain a concise package overview and quick-start guide.

## Coding guidelines

Follow these conventions:

- Prefer clear, explicit Python code over unnecessary abstraction.
- Keep estimator interfaces close to scikit-learn conventions: `fit`, `predict`, `predict_proba` where applicable.
- Validate user inputs where possible.
- Avoid changing public APIs without documenting the change.
- Keep new components modular and registered through the existing factory system when applicable.
- Add docstrings for public classes and methods.
- Use reproducible examples with `random_state` where appropriate.

## Adding a new component

The package uses registry-based factories. New components should be added by subclassing the appropriate base class and registering the implementation under a clear string name.

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

When adding a new component, include:

1. implementation code,
2. unit tests,
3. documentation or examples,
4. notes about valid input domains if the component has domain restrictions.

## Pull request process

Before opening a pull request:

1. Pull the latest `main` branch.
2. Run tests locally.
3. Update documentation if the public behavior changes.
4. Keep the pull request focused on one topic.
5. Fill in the pull request template.

Recommended branch names:

```text
fix/kfc-cobra-params
feature/new-kernel
 docs/api-reference-update
```

Recommended commit message style:

```text
Fix parameter forwarding to COBRA combiners
Add documentation for GradientCOBRA options
Add tests for local model factory registration
```

## Reporting bugs

Use the bug report issue template and include:

- package version,
- Python version,
- operating system,
- minimal reproducible example,
- full traceback,
- expected behavior,
- actual behavior.

## Requesting features

Use the feature request issue template and describe:

- the use case,
- the proposed API,
- expected behavior,
- alternatives considered,
- whether you are willing to submit a pull request.

## Security issues

Do not open a public issue for security vulnerabilities. Follow the instructions in `SECURITY.md`.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License used by this repository.
