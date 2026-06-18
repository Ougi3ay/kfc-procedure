# Installation

## Requirements

`kfc-procedure` requires Python `>= 3.11`.

Core dependencies are `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `xgboost`.

## Create a virtual environment

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows PowerShell"

    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

## Install from PyPI

```bash
pip install kfc-procedure
```

Verify:

```python
import kfc_procedure
print("installed")
```

## Install with COBRA support

```bash
pip install "kfc-procedure[cobra]"
```

## Install from source

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
python -m pip install -e ".[dev,cobra]"
```

## Documentation dependencies

```bash
python -m pip install mkdocs mkdocs-material pymdown-extensions "mkdocstrings[python]"
python -m mkdocs serve -f mkdocs.yml
```

## Common errors

??? failure "`ModuleNotFoundError: No module named 'kfc_procedure'`"
    Make sure the package is installed in the active Python environment:

    ```bash
    python -m pip show kfc-procedure
    python -m pip install kfc-procedure
    ```

??? warning "Wrong import name"
    Use `import kfc_procedure`, not `import kfc-procedure`.
