# Developer Overview

This section explains how to maintain, extend, and document `kfc-procedure`.

## Developer architecture

```mermaid
flowchart TB
    Public["kfc.py<br/>KFCProcedure / KFCRegressor / KFCClassifier"]
    Public --> Steps["core.steps"]
    Steps --> K["KStep"]
    Steps --> F["FStep"]
    Steps --> C["CStep"]
    K --> Div["BregmanDivergenceFactory"]
    F --> LM["LocalModelFactory"]
    C --> Comb["CombinerFactory"]
    Comb --> Cobra["cobra package"]
```

## Development setup

```bash
git clone https://github.com/Ougi3ay/kfc-procedure.git
cd kfc-procedure
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,cobra]"
pytest
```

## Design pattern

The codebase is built around registry factories. New components are usually added by:

1. subclassing a base class,
2. decorating the class with a factory registration,
3. importing the module so registration executes,
4. using the registered string in `KFCProcedure`.
