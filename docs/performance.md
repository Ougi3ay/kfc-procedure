# Performance Considerations


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## KFCProcedure complexity

Let:

- `n` be the number of samples;
- `d` be the number of features;
- `M` be the number of divergences;
- `K` be the number of clusters;
- `T` be the number of clustering iterations;
- `C_fit` be the cost of fitting one local model.

The K-step fits `M` Bregman K-Means models. A typical Lloyd iteration computes distances from each sample to each centroid, giving an approximate cost of:

\[
O(M \cdot T \cdot n \cdot K \cdot d).
\]

The F-step trains up to `M × K` local models, so its cost depends on the selected estimator and the distribution of samples across clusters.

The C-step cost depends on the combiner. Simple mean and majority vote are linear in the prediction matrix size. Stacking and COBRA-based combiners have additional training or distance-matrix costs.

## COBRA distance matrices

COBRA-style estimators compute pairwise distances in prediction space and/or input space. If the calibration set has `m` samples, a full distance matrix can require:

\[
O(m^2)
\]

memory and computation. This is the main scalability bottleneck for large datasets.

## Practical optimization strategies

| Issue | Strategy |
|---|---|
| Slow KFC clustering | Reduce `n_clusters`, reduce number of divergences, reduce `max_iter`, start with `euclidean`. |
| Slow COBRA optimization | Reduce `max_iter`, provide a shorter `bandwidth_list`, reduce `n_cv`, or use fewer base estimators. |
| High memory usage | Use smaller calibration splits or precomputed prediction features with fewer columns. |
| Domain errors | Scale data appropriately before using `gkl`, `is`, or `logistic`. |
| Slow notebooks | Use small synthetic data for demonstration; run full experiments separately. |

## FAISS support

`CombinedClassifierFast` checks for optional `faiss`. The package lists `faiss-cpu` in optional/development dependencies. Use the fast class only when FAISS is installed and approximate neighbor behavior is acceptable for the experiment.
