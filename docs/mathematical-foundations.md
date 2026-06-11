# Mathematical Foundations

!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.

## Bregman divergence

For a strictly convex differentiable generator function \(\phi\), the Bregman divergence between two points \(x\) and \(y\) is:

\[
D_\phi(x, y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x - y \rangle.
\]

The implementation represents this structure through divergence classes that define methods such as `phi`, `grad_phi`, `distance`, `pairwise`, and `in_domain`.

## Implemented divergences

| Factory name | Divergence | Domain |
|---|---|---|
| `euclidean` | squared Euclidean divergence | real-valued arrays |
| `gkl` | generalized Kullback-Leibler divergence | positive arrays |
| `is` | Itakura-Saito divergence | positive arrays |
| `logistic` | logistic/Bernoulli divergence | values in `(0, 1)` |

## Clusterwise prediction

For each divergence \(d_m\), the K-step fits a clustering model:

\[
C_m = \operatorname{BregmanKMeans}_{d_m}(X).
\]

For every cluster \(k\) under divergence \(m\), the F-step trains a local model:

\[
f_{m,k}: X_{m,k} \rightarrow y_{m,k}.
\]

At prediction time, one prediction is produced per divergence. These predictions form a matrix:

\[
P = [\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(M)}] \in \mathbb{R}^{n \times M}.
\]

The C-step applies an aggregation function:

\[
\hat{y} = g(P).
\]

## COBRA-style kernel aggregation

COBRA estimators operate in a prediction space. Let \(z(x)\) denote the vector of base-estimator predictions for sample \(x\), and let \(z_i\) be the prediction-space vector for a calibration sample. A distance function computes:

\[
d_i(x) = d(z(x), z_i).
\]

A kernel adapter applies tunable parameters, and a kernel converts the adapted distance into a weight:

\[
w_i(x) = K(a \cdot d_i(x)).
\]

For regression with weighted mean aggregation:

\[
\hat{y}(x) = \frac{\sum_i w_i(x)y_i}{\sum_i w_i(x)}.
\]

For classification with weighted vote aggregation:

\[
\hat{c}(x) = \arg\max_c \sum_i w_i(x)\mathbf{1}(y_i=c).
\]
