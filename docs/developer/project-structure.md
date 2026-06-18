# Project Structure

Simplified source tree:

```text
kfc-procedure/
├── pyproject.toml
├── README.md
├── src/
│   └── kfc_procedure/
│       ├── __init__.py
│       ├── kfc.py
│       ├── core/
│       │   ├── factory.py
│       │   ├── steps/
│       │   │   ├── kstep.py
│       │   │   ├── fstep.py
│       │   │   └── cstep.py
│       │   ├── clustering/
│       │   │   ├── bregman.py
│       │   │   └── divergences/
│       │   ├── ml/
│       │   └── combiner/
│       ├── cobra/
│       └── utils/
└── tests/
    └── cobra/
```

## Important files

| Path | Purpose |
|---|---|
| `src/kfc_procedure/kfc.py` | public KFC estimator and wrappers |
| `core/steps/kstep.py` | fits one `BregmanKMeans` per divergence |
| `core/steps/fstep.py` | trains local models for each divergence/cluster |
| `core/steps/cstep.py` | builds and fits final combiner |
| `core/clustering/bregman.py` | Lloyd-style Bregman K-Means |
| `core/clustering/divergences/` | divergence implementations and factory |
| `core/ml/sklearn.py` | sklearn local model adapter and auto-registration |
| `core/combiner/` | final aggregation strategies |
| `cobra/` | standalone COBRA estimators |
