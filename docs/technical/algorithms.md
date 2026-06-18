# Algorithms

This page gives step-by-step explanations, pseudocode, input/output definitions, and complexity estimates for the main algorithms.

Notation used in this page:

| Symbol | Meaning |
|---|---|
| `n` | number of samples |
| `d` | number of features |
| `m` | number of divergences |
| `K` | number of clusters per divergence |
| `I` | number of Lloyd iterations |
| `R` | number of random initializations in Bregman K-Means |
| `q` | number of base estimators in COBRA |
| `n_l` | number of calibration samples |

---

## 1. Bregman K-Means

### Input

| Name | Shape | Description |
|---|---:|---|
| `X` | `(n, d)` | Feature matrix. |
| `n_clusters` | scalar | Number of clusters. |
| `divergence` | object | Bregman divergence implementation. |
| `max_iter` | scalar | Maximum Lloyd iterations. |
| `n_init` | scalar | Number of random initializations. |

### Output

| Name | Shape | Description |
|---|---:|---|
| `labels_` | `(n,)` | Cluster assignment for each sample. |
| `cluster_centers_` | `(K, d)` | Learned centroids. |
| `inertia_` | scalar | Best average distortion. |
| `n_iter_` | scalar | Iterations used by best run. |

### Pseudocode

```text
Algorithm: BregmanKMeans.fit(X)
Input: X, divergence D_phi, K, max_iter, n_init
Output: labels, centroids

1. Validate X and divergence domain.
2. best_distortion ← +∞
3. For r = 1 to n_init:
      a. Initialize K centroids by sampling training points.
      b. previous_distortion ← +∞
      c. For t = 1 to max_iter:
            i.   Compute distance matrix D[i, k] = D_phi(X_i, μ_k)
            ii.  Assign label_i = argmin_k D[i, k]
            iii. Update each centroid μ_k = mean({X_i : label_i = k})
            iv.  Reinitialize empty clusters with random data points.
            v.   Compute average minimum distortion.
            vi.  Stop if relative distortion change < tol.
      d. If run distortion is lower than best_distortion, save run.
4. Store best labels and centroids.
```

### Complexity

Each Lloyd iteration computes a distance matrix between `n` samples and `K` centroids.

| Quantity | Complexity |
|---|---:|
| Time per iteration | `O(n K d)` |
| Time per fit | `O(R I n K d)` |
| Distance matrix memory | `O(n K)` |
| Centroid memory | `O(K d)` |

The implementation computes distortion in blocks to reduce peak memory during scoring, but assignment still builds a distance matrix for each iteration.

---

## 2. K-step: multi-divergence clustering

### Input

| Name | Shape | Description |
|---|---:|---|
| `X` | `(n, d)` | Training features. |
| `divergences` | list length `m` | Divergence identifiers or objects. |
| `n_clusters` | scalar | Number of clusters per divergence. |

### Output

```python
models_ = {
    "euclidean": BregmanKMeans(...),
    "gkl": BregmanKMeans(...),
}

clusters_ = {
    "euclidean": array([...]),
    "gkl": array([...]),
}
```

### Pseudocode

```text
Algorithm: KStep.fit(X)
Input: X, divergences = [d1, ..., dm]
Output: models_, clusters_

1. Convert X to float array.
2. For each divergence d_j:
      a. Resolve string name to divergence object.
      b. Create BregmanKMeans(n_clusters=K, divergence=d_j).
      c. Fit BregmanKMeans on X.
      d. Store fitted model under divergence name.
      e. Store model.labels_ under divergence name.
3. Return self.
```

### Complexity

If each divergence uses `R` initializations and `I` iterations:

| Quantity | Complexity |
|---|---:|
| Time | `O(m R I n K d)` |
| Memory | `O(m (K d + n))` for stored models and cluster labels |

---

## 3. F-step: cluster-local model fitting

### Input

| Name | Shape | Description |
|---|---:|---|
| `X` | `(n, d)` | Training features. |
| `y` | `(n,)` | Target values or class labels. |
| `clusters` | dictionary | One cluster assignment array per divergence. |
| `local_model` | string or object | Local supervised model. |

### Output

Nested dictionary of fitted local models:

```python
models_[divergence_name][cluster_key] = {
    "divergence": div_name,
    "cluster": cluster_id,
    "model": fitted_model,
}
```

### Pseudocode

```text
Algorithm: FStep.fit(X, y, clusters)
Input: X, y, clusters dictionary
Output: models_

1. For each divergence name div_name:
      a. Create empty model dictionary.
      b. For each cluster id k in unique cluster labels:
            i.   Select samples where clusters[div_name] == k.
            ii.  Resolve local model from LocalModelFactory.
            iii. Fit local model on selected samples.
            iv.  Store fitted model.
2. Return self.
```

### Prediction pseudocode

```text
Algorithm: FStep.predict(X, clusters)
Input: X, cluster assignments for X
Output: P prediction matrix

1. outputs ← []
2. For each divergence div_name:
      a. pred ← array of NaN with length n
      b. For each fitted cluster model:
            i. Select samples assigned to this cluster.
            ii. Predict those samples using the cluster model.
            iii. Write predictions into pred.
      c. Append pred as one column to outputs.
3. Return column_stack(outputs)
```

### Complexity

Let `T_fit(n_k, d)` be the training cost of the selected local model on cluster `k`, and `T_pred(n_k, d)` its prediction cost.

| Quantity | Complexity |
|---|---:|
| Fit time | `O(Σ_j Σ_k T_fit(n_{j,k}, d))` |
| Predict time | `O(Σ_j Σ_k T_pred(n_{j,k}^{test}, d))` |
| Model memory | `O(m K × model_size)` |
| Prediction matrix memory | `O(n m)` |

---

## 4. C-step: prediction aggregation

### Input

| Name | Shape | Description |
|---|---:|---|
| `P` | `(n, m)` | Prediction matrix from F-step. |
| `y` | `(n,)` | Ground-truth values for training learned combiners. |
| `combiner` | string or object | Aggregation strategy. |

### Output

| Name | Shape | Description |
|---|---:|---|
| `y_pred` | `(n,)` | Final prediction. |

### Pseudocode

```text
Algorithm: CStep.fit(P, y)
Input: prediction matrix P, target y
Output: fitted strategy_

1. Resolve combiner from CombinerFactory.
2. Fit combiner on (P, y).
3. Store fitted strategy_.
```

```text
Algorithm: CStep.predict(P)
Input: prediction matrix P
Output: final predictions

1. Check that strategy_ is fitted.
2. Return strategy_.predict(P).
```

### Complexity by combiner

| Combiner | Fit time | Predict time | Notes |
|---|---:|---:|---|
| `mean` | `O(1)` | `O(n m)` | Stateless row mean. |
| `weighted_mean` | approximately `O(n m^2 + m^3)` | `O(n m)` | Linear regression over prediction columns. |
| `stacking_regressor` | depends on meta-model | depends on meta-model | Default is linear regression. |
| `majority_vote` | `O(1)` | `O(n m)` | Row-wise mode. |
| `stacking_classifier` | depends on meta-classifier | depends on meta-classifier | Default logistic regression. |
| COBRA combiners | `O(n_l^2 m + optimization)` | `O(n_test n_l m)` | Kernel aggregation over calibration samples. |

---

## 5. Full KFCProcedure training algorithm

### Input

| Name | Shape | Description |
|---|---:|---|
| `X` | `(n, d)` | Feature matrix. |
| `y` | `(n,)` | Target values or labels. |
| `divergences` | list length `m` | Divergences for K-step. |
| `local_model` | string/object | F-step local model. |
| `combiner` | string/object | C-step combiner. |

### Output

A fitted estimator containing:

- `kstep_`
- `fstep_`
- `cstep_`

### Pseudocode

```text
Algorithm: KFCProcedure.fit(X, y)
Input: X, y
Output: fitted KFCProcedure

1. Convert X and y to NumPy arrays.
2. Split data into two halves:
      (X_k, y_k) for K-step and F-step
      (X_l, y_l) for C-step calibration
   If task is classification, stratify by y.
3. K-step:
      a. Fit KStep on X_k.
      b. Obtain training clusters for X_k.
      c. Predict clusters for X_l.
4. F-step:
      a. Fit local models on X_k, y_k using training clusters.
      b. Predict X_l using local models and X_l clusters.
      c. Build prediction matrix P_l.
5. C-step:
      a. Fit combiner on P_l, y_l.
6. Return fitted estimator.
```

### Prediction pseudocode

```text
Algorithm: KFCProcedure.predict(X_new)
Input: X_new
Output: final predictions

1. Predict cluster assignments for X_new using kstep_.
2. Predict local outputs using fstep_.
3. Build prediction matrix P.
4. Return cstep_.predict(P).
```

### Complexity

Let `n_t ≈ n/2` be the K/F training size and `n_l ≈ n/2` be the C-step calibration size.

| Stage | Time |
|---|---:|
| K-step fit | `O(m R I n_t K d)` |
| F-step fit | `O(Σ_j Σ_k T_fit(n_{j,k}, d))` |
| F-step calibration prediction | `O(Σ_j Σ_k T_pred(n_{j,k}^{cal}, d))` |
| C-step fit | combiner-dependent |

Prediction for `s` test samples:

| Stage | Time |
|---|---:|
| Cluster assignment | `O(m s K d)` |
| Local prediction | `O(Σ_j Σ_k T_pred(s_{j,k}, d))` |
| Aggregation | combiner-dependent |

---

## 6. GradientCOBRA algorithm

### Input

| Name | Shape | Description |
|---|---:|---|
| `X` | `(n, d)` | Training features or prediction matrix if `as_predictions=True`. |
| `y` | `(n,)` | Regression target. |
| `estimators` | list | Base estimators, unless using precomputed predictions. |

### Output

- fitted base estimators, if not using `as_predictions=True`;
- calibration prediction matrix;
- optimized bandwidth;
- fitted distance, kernel, loss, optimizer, and aggregator components.

### Pseudocode

```text
Algorithm: GradientCOBRA.fit(X, y)
Input: X, y
Output: fitted GradientCOBRA

1. Resolve training context into model-training and calibration subsets.
2. If as_predictions is False:
      a. Fit base estimators on X_k, y_k.
      b. Predict calibration set X_l to build Z_l.
   Else:
      a. Treat X_l as precomputed prediction features Z_l.
3. Normalize prediction matrix Z_l.
4. Compute pairwise distance matrix between calibration prediction vectors.
5. Create cross-validation folds on the calibration set.
6. Optimize bandwidth by minimizing CV loss:
      a. Transform distances with bandwidth.
      b. Apply kernel to obtain weights.
      c. Aggregate calibration targets.
      d. Evaluate loss.
7. Store best bandwidth and optimization history.
```

### Prediction pseudocode

```text
Algorithm: GradientCOBRA.predict(X_new)
Input: X_new
Output: y_pred

1. Build prediction-space representation Z_new.
2. Normalize Z_new using stored normalization constant.
3. Compute distances from Z_new to calibration prediction matrix Z_l.
4. Apply optimized bandwidth and kernel.
5. Aggregate calibration targets y_l using kernel weights.
6. Return predictions.
```

### Complexity

| Stage | Complexity |
|---|---:|
| Base estimator training | `Σ T_fit_base` |
| Calibration predictions | `Σ T_pred_base` |
| Calibration distance matrix | `O(n_l^2 q)` |
| Grid optimization | `O(G n_l^2 + CV aggregation)` |
| Prediction | `O(s n_l q)` plus base estimator prediction |

`G` is the number of bandwidth candidates.

---

## 7. MixCOBRA algorithm

`MixCOBRARegressor` extends COBRA by combining two similarity spaces:

- original input space `X`;
- prediction space `Z` from base estimator outputs.

### Pseudocode

```text
Algorithm: MixCOBRARegressor.fit(X, y)
Input: X, y
Output: fitted MixCOBRARegressor

1. Resolve training and calibration subsets.
2. Train base estimators and compute prediction matrix Z_l, unless as_predictions=True.
3. Normalize X_l and Z_l.
4. If one_parameter=True:
      a. Concatenate normalized [X_l, Z_l].
      b. Compute one mixed distance matrix.
      c. Optimize a bandwidth-like parameter.
5. Else:
      a. Compute D_x in input space.
      b. Compute D_z in prediction space.
      c. Optimize alpha and beta in D = alpha D_x + beta D_z.
6. Store optimized parameters.
```

### Prediction pseudocode

```text
Algorithm: MixCOBRARegressor.predict(X_new)
Input: X_new
Output: predictions

1. Generate or receive prediction features Z_new.
2. Normalize X_new and Z_new.
3. Compute distances to calibration samples.
4. Combine distances using optimized parameter(s).
5. Apply kernel.
6. Aggregate calibration targets with kernel weights.
```

### Complexity

| Stage | Complexity |
|---|---:|
| Fit distance matrices | `O(n_l^2 d + n_l^2 q)` |
| Grid optimization, two parameters | `O(G_alpha G_beta n_l^2)` |
| Grid optimization, one parameter | `O(G n_l^2)` |
| Prediction | `O(s n_l (d + q))` |

---

## 8. CombinedClassifier algorithm

`CombinedClassifier` is the classification counterpart to kernel-weighted COBRA aggregation.

### Pseudocode

```text
Algorithm: CombinedClassifier.fit(X, y)
Input: X, class labels y
Output: fitted CombinedClassifier

1. Resolve training and calibration subsets.
2. Train base estimators and compute calibration prediction matrix, unless as_predictions=True.
3. Store class labels and global majority class.
4. Resolve distance, kernel, aggregator, loss, optimizer, adapter, and CV components.
5. Compute pairwise distances in prediction space.
6. Optimize kernel bandwidth using cross-validation.
```

```text
Algorithm: CombinedClassifier.predict(X_new)
Input: X_new
Output: predicted class labels

1. Compute prediction-space representation for X_new.
2. Compute distances to calibration prediction matrix.
3. Apply bandwidth and kernel to obtain weights.
4. If no positive weights, return the global majority class.
5. Otherwise, apply weighted vote over calibration labels.
```

### Complexity

| Stage | Complexity |
|---|---:|
| Calibration distance matrix | `O(n_l^2 q)` |
| Optimization | `O(G n_l^2)` for grid search |
| Prediction | `O(s n_l q)` |
| Probability prediction | `O(s n_l + s C)` where `C` is number of classes |
