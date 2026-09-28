# Installation

KFC Procedure requires Python 3.9 or newer.

## Install with pip

Install the latest release from PyPI:

```bash
pip install kfc-procedure
```

Verify the installation:

```bash
python -c "import kfc_procedure; print('KFC Procedure installed successfully')"
```

## Install from source

Clone the repository:

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
```

Install the package in editable mode:

```bash
pip install -e .
```

Editable mode is recommended for development because changes to the source code
are immediately available without reinstalling the package.

## Development dependencies

If the project defines development extras, install them with:

```bash
pip install -e ".[dev]"
```

For documentation development:

```bash
pip install -e ".[docs]"
```

Then start the documentation server:

```bash
mkdocs serve
```

## Verify the package

Open Python and import the package:

```python
import kfc_procedure
```

You can also import the main interface:

```python
from kfc_procedure import KFC
```
