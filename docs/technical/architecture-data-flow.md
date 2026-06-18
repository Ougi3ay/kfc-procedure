# Architecture & Data Flow

This page explains how data moves through the package during training and prediction.

---

## 1. Package architecture

```mermaid
flowchart TD
    subgraph PublicAPI["Public API"]
        A1["kfc_procedure.__init__"]
        A2["KFCProcedure"]
        A3["KFCRegressor"]
        A4["KFCClassifier"]
        A5["COBRA estimators"]
    end

    subgraph KFC["KFC Core"]
        K1["KStep"]
        K2["FStep"]
        K3["CStep"]
    end

    subgraph Clustering["Clustering"]
        B1["BregmanKMeans"]
        B2["Bregman divergences"]
    end

    subgraph ML["Local Models"]
        M1["LocalModelFactory"]
        M2["SklearnLocalModel"]
        M3["MeanRegressor"]
    end

    subgraph Combiner["Combiners"]
        C1["CombinerFactory"]
        C2["Regression combiners"]
        C3["Classification combiners"]
        C4["COBRA wrappers"]
    end

    subgraph CobraCore["COBRA Core"]
        D1["Distances"]
        D2["Kernels"]
        D3["Adapters"]
        D4["Aggregators"]
        D5["Losses"]
        D6["Optimizers"]
        D7["CV"]
    end

    PublicAPI --> KFC
    K1 --> Clustering
    K2 --> ML
    K3 --> Combiner
    C4 --> CobraCore
    A5 --> CobraCore
```

---

## 2. KFCProcedure training data flow

`KFCProcedure.fit(X, y)` uses an internal split:

```mermaid
flowchart LR
    Input["X, y"] --> Split["train_test_split<br/>50% K/F training<br/>50% C calibration"]

    Split --> XK["X_k, y_k"]
    Split --> XL["X_l, y_l"]

    XK --> KStep["KStep.fit(X_k)"]
    KStep --> ClustersK["clusters_k"]
    KStep --> PredictXL["KStep.predict(X_l)"]
    PredictXL --> ClustersL["clusters_l"]

    XK --> FStep["FStep.fit(X_k, y_k, clusters_k)"]
    ClustersK --> FStep

    XL --> FPred["FStep.predict(X_l, clusters_l)"]
    ClustersL --> FPred
    FStep --> FPred

    FPred --> PL["Prediction matrix P_l"]
    PL --> CStep["CStep.fit(P_l, y_l)"]
    XL --> CStep

    CStep --> Model["Fitted KFCProcedure"]
```

---

## 3. KFCProcedure prediction data flow

```mermaid
flowchart LR
    XNew["X_new"] --> Cluster["kstep_.predict(X_new)"]
    Cluster --> Assignments["Cluster assignments per divergence"]
    XNew --> FPred["fstep_.predict(X_new, assignments)"]
    Assignments --> FPred
    FPred --> P["Prediction matrix P"]
    P --> Combine["cstep_.predict(P)"]
    Combine --> Yhat["Final predictions"]
```

---

## 4. Object lifecycle

```mermaid
sequenceDiagram
    participant User
    participant KFC as KFCProcedure
    participant K as KStep
    participant F as FStep
    participant C as CStep

    User->>KFC: fit(X, y)
    KFC->>KFC: split X,y into X_k/y_k and X_l/y_l
    KFC->>K: fit(X_k)
    K-->>KFC: clusters_k
    KFC->>K: predict(X_l)
    K-->>KFC: clusters_l
    KFC->>F: fit(X_k, y_k, clusters_k)
    KFC->>F: predict(X_l, clusters_l)
    F-->>KFC: P_l
    KFC->>C: fit(P_l, y_l)
    C-->>KFC: fitted combiner
    KFC-->>User: self

    User->>KFC: predict(X_new)
    KFC->>K: predict(X_new)
    K-->>KFC: clusters_new
    KFC->>F: predict(X_new, clusters_new)
    F-->>KFC: P_new
    KFC->>C: predict(P_new)
    C-->>KFC: y_pred
    KFC-->>User: y_pred
```

---

## 5. COBRA training data flow

The COBRA estimators follow a separate but related calibration architecture.

```mermaid
flowchart TD
    Input["X, y"] --> Context["resolve_training_context"]
    Context --> Train["X_k, y_k<br/>base estimator training"]
    Context --> Calib["X_l, y_l<br/>calibration set"]

    Train --> BaseModels["Fit base estimators"]
    BaseModels --> PredCalib["Predict X_l"]
    Calib --> PredCalib

    PredCalib --> ZL["Prediction space Z_l"]
    ZL --> Normalize["Normalize prediction space"]
    Normalize --> Dist["Pairwise distance matrix"]
    Dist --> KernelAdapter["Kernel adapter<br/>bandwidth / alpha-beta"]
    KernelAdapter --> Kernel["Kernel weights"]
    Kernel --> CV["Cross-validation loss"]
    CV --> Optimizer["Optimizer selects parameters"]
    Optimizer --> Fitted["Fitted COBRA estimator"]
```

---

## 6. COBRA prediction data flow

```mermaid
flowchart LR
    XNew["X_new"] --> Pred["Base estimators predict"]
    Pred --> ZNew["Prediction vector Z_new"]
    ZNew --> Dist["Distance to calibration Z_l"]
    Dist --> Adapter["Apply learned bandwidth / mix params"]
    Adapter --> Kernel["Kernel weights"]
    Kernel --> Agg["Aggregate y_l"]
    Agg --> Yhat["Prediction"]
```

---

## 7. Registry flow

```mermaid
flowchart LR
    Config["String name in user config"] --> Factory["Factory.contains / create"]
    Factory --> Registry["Internal registry"]
    Registry --> Class["Implementation class"]
    Class --> Instance["Component instance"]
    Instance --> Pipeline["Used by pipeline"]
```

Example:

```python
CombinerFactory.create("weighted_mean")
```

resolves to:

```text
WeightedMeanCombiner(...)
```

---

## 8. Data structures

### K-step outputs

```python
models_ = {
    "euclidean": BregmanKMeans(...),
    "gkl": BregmanKMeans(...),
}

clusters_ = {
    "euclidean": np.ndarray(shape=(n_samples,)),
    "gkl": np.ndarray(shape=(n_samples,)),
}
```

### F-step outputs

```python
models_ = {
    "euclidean": {
        "m0": {"divergence": "euclidean", "cluster": 0, "model": ...},
        "m1": {"divergence": "euclidean", "cluster": 1, "model": ...},
    }
}
```

### F-step prediction matrix

```text
P.shape = (n_samples, n_divergences)
```

Each column corresponds to one divergence view.

---

## 9. Component boundaries

| Boundary | Input | Output |
|---|---|---|
| `BregmanKMeans` | `X` | cluster labels and centroids |
| `KStep` | `X` and divergences | dictionary of cluster labels |
| `FStep` | `X`, `y`, clusters | local models; prediction matrix |
| `CStep` | prediction matrix, `y` | fitted combiner; final predictions |
| `GradientCOBRA` | `X`, `y` | kernel aggregation model |
| `MixCOBRARegressor` | `X`, `y` | mixed-distance aggregation model |
| `CombinedClassifier` | `X`, `y` | kernel voting classifier |
