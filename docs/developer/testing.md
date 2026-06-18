# Testing

The repository includes pytest configuration in `pyproject.toml`:

```toml
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-vv --disable-warnings --strict-markers"
```

## Run tests

```bash
python -m pip install -e ".[dev,cobra]"
pytest
```

## Run only COBRA tests

```bash
pytest tests/cobra
```

## Suggested additional tests

The current tests focus heavily on COBRA components. Useful future tests include:

- `KFCRegressor.fit/predict` smoke tests,
- `KFCClassifier.fit/predict` smoke tests,
- divergence domain validation tests,
- empty-cluster behavior tests,
- C-step combiner compatibility tests,
- serialization tests with `joblib`.
