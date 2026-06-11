# Complete Generated API Inventory


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.

This page lists public modules, classes, methods, and functions extracted from the implementation.

## `kfc_procedure.cobra.combined_classifier`

### `CombinedClassifier`

```python
CombinedClassifier(self, estimators: 'List[Union[str, BaseEstimator]] | None' = None, estimators_params: 'Dict[str, Any] | None' = None, distance: 'str' = 'hamming', distance_params: 'Dict[str, Any] | None' = None, kernel: 'str' = 'rbf', kernel_params: 'Dict[str, Any] | None' = None, aggregator: 'str' = 'weighted_vote', aggregator_params: 'Dict[str, Any] | None' = None, loss: 'str' = 'mse', loss_params: 'dict[str, Any] | None' = None, optimizer: 'str' = 'grid', optimizer_params: 'dict[str, Any] | None' = None, n_jobs: 'int' = 1, bandwidth_list: 'np.ndarray | None' = None, max_iter: 'int' = 300, n_cv: 'int' = 5, random_state: 'int | None' = None)
```

CombineClassifier
=================

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X, y, X_l=None, y_l=None, split_ratio=0.5, overlap=False, as_predictions=False)` |
| `get_params` | `(self, deep=True)` |
| `kappa_cross_validation_error` | `(self, params)` |
| `predict` | `(self, X)` |
| `predict_proba` | `(self, X)` |
| `set_params` | `(self, **params)` |

### `CombinedClassifierFast`

```python
CombinedClassifierFast(self, use_faiss: 'bool' = False, faiss_k: 'int | None' = None, **kwargs)
```

CombineClassifier
=================

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X, y, X_l=None, y_l=None, split_ratio=0.5, overlap=False, as_predictions=False)` |
| `get_params` | `(self, deep=True)` |
| `kappa_cross_validation_error` | `(self, params)` |
| `predict` | `(self, X)` |
| `predict_proba` | `(self, X, pred_X)` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.cobra.core.adapters.base`

### `BaseKernelAdapter`

```python
BaseKernelAdapter(self, **kwargs)
```

Abstract base class for kernel transformation adapters.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `parameter_vector` | `(self) -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |
| `transform` | `(self, *distances: 'np.ndarray') -> 'np.ndarray'` |

### `KernelAdapterFactory`

```python
KernelAdapterFactory(self, /, *args, **kwargs)
```

Factory for kernel adapters.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.adapters.one_parameter`

### `OneParameterKernelAdapter`

```python
OneParameterKernelAdapter(self, bandwidth: 'float' = 1.0)
```

Simple scaling kernel adapter.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `parameter_vector` | `(self) -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |
| `transform` | `(self, *distances: 'np.ndarray') -> 'np.ndarray'` |

## `kfc_procedure.cobra.core.adapters.two_parameter`

### `TwoParameterKernelAdapter`

```python
TwoParameterKernelAdapter(self, alpha: 'float' = 1.0, beta: 'float' = 0.0)
```

Linear combination kernel adapter.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `parameter_vector` | `(self) -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |
| `transform` | `(self, *distances: 'np.ndarray') -> 'np.ndarray'` |

## `kfc_procedure.cobra.core.aggregators.base`

### `AggregatorFactory`

```python
AggregatorFactory(self, /, *args, **kwargs)
```

Abstract registry-based factory.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

### `BaseAggregator`

```python
BaseAggregator(self, /, *args, **kwargs)
```

Base class for COBRA aggregation strategies.

#### Methods

| Method | Signature |
|---|---|
| `aggregate` | `(self, values: 'np.ndarray', weights: 'np.ndarray \| None' = None, **kwargs)` |
| `aggregate_matrix` | `(self, values: 'np.ndarray', weights: 'np.ndarray', fallback: 'float \| None' = None, **kwargs) -> 'np.ndarray'` |
| `aggregate_proba` | `(self, values: 'np.ndarray', weights: 'np.ndarray \| None' = None, classes: 'np.ndarray \| None' = None, **kwargs)` |

## `kfc_procedure.cobra.core.aggregators.weighted_mean`

### `WeightedMeanAggregator`

```python
WeightedMeanAggregator(self, /, *args, **kwargs)
```

Weighted mean aggregation (regression).

#### Methods

| Method | Signature |
|---|---|
| `aggregate` | `(self, values, weights=None, fallback=None, **kwargs)` |
| `aggregate_matrix` | `(self, values: 'np.ndarray', weights: 'np.ndarray', fallback: 'float \| None' = None, **kwargs) -> 'np.ndarray'` |
| `aggregate_proba` | `(self, values, weights=None, classes=None, **kwargs)` |

## `kfc_procedure.cobra.core.aggregators.weighted_vote`

### `WeightedVoteAggregator`

```python
WeightedVoteAggregator(self, /, *args, **kwargs)
```

Fully vectorized weighted majority vote.

#### Methods

| Method | Signature |
|---|---|
| `aggregate` | `(self, values, weights=None, **kwargs)` |
| `aggregate_matrix` | `(self, values, weights, **kwargs)` |
| `aggregate_proba` | `(self, values, weights=None, classes=None, **kwargs)` |
| `aggregate_proba_batch` | `(self, values, weights, classes=None, **kwargs)` |

## `kfc_procedure.cobra.core.cv.base`

### `BaseCrossValidator`

```python
BaseCrossValidator(self, /, *args, **kwargs)
```

Abstract base class for cross-validation strategies.

#### Methods

| Method | Signature |
|---|---|
| `get_n_splits` | `(self) -> 'int'` |
| `split` | `(self, x: 'ArrayLike', y: 'ArrayLike', *, groups: 'ArrayLike \| None' = None) -> 'Iterator[SplitIndices]'` |

### `CVFactory`

```python
CVFactory(self, /, *args, **kwargs)
```

Factory for cross-validation strategies.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.cv.kfold`

### `KFoldCV`

```python
KFoldCV(self, n_splits: 'int' = 5, shuffle: 'bool' = True, random_state: 'int | None' = None)
```

K-Fold Cross-Validation.

#### Methods

| Method | Signature |
|---|---|
| `get_n_splits` | `(self, x: 'ArrayLike \| None' = None, y: 'ArrayLike \| None' = None) -> 'int'` |
| `split` | `(self, x: 'ArrayLike', y: 'ArrayLike') -> 'Iterator[SplitIndices]'` |

## `kfc_procedure.cobra.core.cv.stratified_kfold`

### `StratifiedKFoldCV`

```python
StratifiedKFoldCV(self, n_splits: 'int' = 5, random_state: 'int | None' = None)
```

Stratified K-Fold Cross Validation.

#### Methods

| Method | Signature |
|---|---|
| `get_n_splits` | `(self) -> 'int'` |
| `split` | `(self, x: 'ArrayLike', y: 'ArrayLike')` |

## `kfc_procedure.cobra.core.cv.time_series`

### `TimeSeriesCV`

```python
TimeSeriesCV(self, n_splits: 'int' = 5, test_size: 'int | None' = None)
```

Time Series Cross Validation.

#### Methods

| Method | Signature |
|---|---|
| `get_n_splits` | `(self) -> 'int'` |
| `split` | `(self, x: 'ArrayLike', y: 'ArrayLike')` |

## `kfc_procedure.cobra.core.distances.base`

### `BaseDistance`

```python
BaseDistance(self, **kwargs)
```

Abstract base class for all distance metrics.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True)` |
| `matrix` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

### `DistanceFactory`

```python
DistanceFactory(self, /, *args, **kwargs)
```

Factory class for distance metrics.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.distances.cosine`

### `CosineDistance`

```python
CosineDistance(self, **kwargs)
```

Cosine distance metric.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True)` |
| `matrix` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.cobra.core.distances.euclidean`

### `EuclideanDistance`

```python
EuclideanDistance(self, **kwargs)
```

Euclidean (L2) distance metric.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True)` |
| `matrix` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.cobra.core.distances.hamming`

### `HammingDistance`

```python
HammingDistance(self, **kwargs)
```

Hamming distance metric.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True)` |
| `matrix` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.cobra.core.distances.manhattan`

### `ManhattanDistance`

```python
ManhattanDistance(self, **kwargs)
```

Manhattan (L1) distance metric.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True)` |
| `matrix` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.cobra.core.distances.minkowski`

### `MinkowskiDistance`

```python
MinkowskiDistance(self, p: 'float' = 3, **kwargs)
```

Minkowski (Lp) distance metric.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True)` |
| `matrix` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.cobra.core.estimators.base`

### `BaseEstimator`

```python
BaseEstimator(self, /, *args, **kwargs)
```

Abstract base class for all COBRA estimators.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, x: 'np.ndarray', y: 'np.ndarray', **kwargs) -> "'BaseEstimator'"` |
| `predict` | `(self, x: 'np.ndarray', **kwargs) -> 'np.ndarray'` |

### `EstimatorFactory`

```python
EstimatorFactory(self, /, *args, **kwargs)
```

Registry-based factory for estimator classes.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.estimators.mean_regressor`

### `MeanRegressor`

```python
MeanRegressor(self) -> 'None'
```

Mean baseline regressor.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, x: 'np.ndarray', y: 'np.ndarray', **kwargs) -> "'MeanRegressor'"` |
| `predict` | `(self, x: 'np.ndarray', **kwargs) -> 'np.ndarray'` |

## `kfc_procedure.cobra.core.estimators.sklearn`

### `SklearnEstimator`

```python
SklearnEstimator(self, estimator_cls: Type[sklearn.base.BaseEstimator], **kwargs)
```

Wrapper for sklearn estimators to make them compatible
with the COBRA BaseEstimator interface.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, x, y)` |
| `get_params` | `(self, deep: bool = True)` |
| `predict` | `(self, x)` |
| `predict_proba` | `(self, x)` |
| `set_params` | `(self, **params)` |

### Functions

| Function | Signature |
|---|---|
| `register_all_sklearn_estimators` | `(factory)` |

## `kfc_procedure.cobra.core.factory`

### `BaseFactory`

```python
BaseFactory(self, /, *args, **kwargs)
```

Abstract registry-based factory.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.kernels.base`

### `BaseKernel`

```python
BaseKernel(self, **kwargs)
```

Abstract base class for kernel functions.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

### `KernelFactory`

```python
KernelFactory(self, /, *args, **kwargs)
```

Factory for kernel functions.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.kernels.biweight`

### `BiweightKernel`

```python
BiweightKernel(self, **kwargs)
```

Biweight kernel (compact support).

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.cauchy`

### `CauchyKernel`

```python
CauchyKernel(self, **kwargs)
```

Cauchy kernel with heavy-tailed decay.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.cobra`

### `COBRAKernel`

```python
COBRAKernel(self, threshold: 'float' = 0.5)
```

Binary threshold kernel.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.epanechnikov`

### `EpanechnikovKernel`

```python
EpanechnikovKernel(self, **kwargs)
```

Compact-support Epanechnikov kernel.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.exponential`

### `ExponentialKernel`

```python
ExponentialKernel(self, exponent: 'float' = 1.0)
```

Exponential decay kernel.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.naive`

### `NaiveKernel`

```python
NaiveKernel(self, **kwargs)
```

Identity kernel (no transformation).

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.radial`

### `RadialKernel`

```python
RadialKernel(self, **kwargs)
```

Radial basis kernel (simplified form).

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.reverse_cosh`

### `ReverseCoshKernel`

```python
ReverseCoshKernel(self, exponent: 'float' = 1.0)
```

Reverse hyperbolic cosine kernel.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.triangular`

### `TriangularKernel`

```python
TriangularKernel(self, **kwargs)
```

Linear triangular kernel.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.kernels.triweight`

### `TriweightKernel`

```python
TriweightKernel(self, **kwargs)
```

Triweight compact kernel.

#### Methods

| Method | Signature |
|---|---|
| `get_params` | `(self, deep: 'bool' = True) -> 'Dict[str, Any]'` |
| `is_continuous` | `(self) -> 'bool'` |
| `is_discrete` | `(self) -> 'bool'` |
| `set_params` | `(self, **params) -> "'BaseKernel'"` |

## `kfc_procedure.cobra.core.losses.base`

### `BaseLoss`

```python
BaseLoss(self, /, *args, **kwargs)
```

Abstract base class for loss functions.

### `LossFactory`

```python
LossFactory(self, /, *args, **kwargs)
```

Factory for loss functions.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.losses.hinge`

### `HingeLoss`

```python
HingeLoss(self, /, *args, **kwargs)
```

Abstract base class for loss functions.

## `kfc_procedure.cobra.core.losses.huber`

### `HuberLoss`

```python
HuberLoss(self, delta: 'float' = 1.0)
```

Abstract base class for loss functions.

## `kfc_procedure.cobra.core.losses.log_loss`

### `LogLoss`

```python
LogLoss(self, /, *args, **kwargs)
```

Abstract base class for loss functions.

## `kfc_procedure.cobra.core.losses.mae`

### `MAELoss`

```python
MAELoss(self, /, *args, **kwargs)
```

Abstract base class for loss functions.

## `kfc_procedure.cobra.core.losses.mse`

### `MSELoss`

```python
MSELoss(self, /, *args, **kwargs)
```

Abstract base class for loss functions.

## `kfc_procedure.cobra.core.losses.quantile`

### `QuantileLoss`

```python
QuantileLoss(self, tau: 'float' = 0.5)
```

Abstract base class for loss functions.

## `kfc_procedure.cobra.core.normalizers.base`

### `BaseNormalizer`

```python
BaseNormalizer(self, /, *args, **kwargs)
```

Abstract base class for all normalization strategies.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, x: 'np.ndarray', **kwargs) -> "'BaseNormalizer'"` |
| `fit_transform` | `(self, x: 'np.ndarray', **kwargs) -> 'np.ndarray'` |
| `transform` | `(self, x: 'np.ndarray', **kwargs) -> 'np.ndarray'` |

### `NormalizerFactory`

```python
NormalizerFactory(self, /, *args, **kwargs)
```

Factory class for normalization strategies.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.normalizers.minmax`

### `MinMaxNormalizer`

```python
MinMaxNormalizer(self)
```

Min-Max normalizer.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, x: numpy.ndarray, **kwargs) -> 'MinMaxNormalizer'` |
| `fit_transform` | `(self, x: 'np.ndarray', **kwargs) -> 'np.ndarray'` |
| `transform` | `(self, x: numpy.ndarray, **kwargs) -> numpy.ndarray` |

## `kfc_procedure.cobra.core.normalizers.standard`

### `StandardNormalizer`

```python
StandardNormalizer(self)
```

Standard (Z-score) normalizer.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, x: numpy.ndarray, **kwargs) -> 'StandardNormalizer'` |
| `fit_transform` | `(self, x: 'np.ndarray', **kwargs) -> 'np.ndarray'` |
| `transform` | `(self, x: numpy.ndarray, **kwargs) -> numpy.ndarray` |

## `kfc_procedure.cobra.core.optimizers._utils`

### Functions

| Function | Signature |
|---|---|
| `central_difference_gradient` | `(objective: 'Callable', params: 'np.ndarray', eps: 'float' = 1e-07) -> 'np.ndarray'` |
| `complex_step_gradient` | `(objective: 'Callable', params: 'np.ndarray', eps: 'float' = 1e-20) -> 'np.ndarray'` |
| `compute_gradient` | `(objective: 'Callable', params: 'np.ndarray', gradient: 'Optional[Callable]' = None, method: 'str' = 'central', eps: 'float' = 1e-07, n_jobs: 'Optional[int]' = None) -> 'np.ndarray'` |
| `forward_difference_gradient` | `(objective: 'Callable', params: 'np.ndarray', eps: 'float' = 1e-07) -> 'np.ndarray'` |
| `parallel_central_difference_gradient` | `(objective: 'Callable', params: 'np.ndarray', eps: 'float' = 1e-07, n_jobs: 'int' = -1) -> 'np.ndarray'` |
| `spsa_gradient` | `(objective: 'Callable', params: 'np.ndarray', eps: 'float' = 1e-07) -> 'np.ndarray'` |

## `kfc_procedure.cobra.core.optimizers.base`

### `BaseOptimizer`

```python
BaseOptimizer(self, show_process: 'bool' = True, **kwargs)
```

Abstract optimizer interface for COBRA.

#### Methods

| Method | Signature |
|---|---|
| `optimize` | `(self, objective: 'Callable[[np.ndarray], float]', init_param: 'np.ndarray \| None' = None, **kwargs) -> 'Dict[str, Any]'` |

### `OptimizerFactory`

```python
OptimizerFactory(self, /, *args, **kwargs)
```

Factory for registering and instantiating optimizers.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.optimizers.gradient.adam`

### `AdamOptimizer`

```python
AdamOptimizer(self, beta1: 'float' = 0.9, beta2: 'float' = 0.999, **kwargs)
```

Adam (Adaptive Moment Estimation) optimizer.

#### Methods

| Method | Signature |
|---|---|
| `gradient` | `(self, objective: 'Callable', params: 'np.ndarray', grad_fn: 'Optional[Callable]' = None) -> 'np.ndarray'` |
| `optimize` | `(self, objective: 'Callable[[np.ndarray], float]', init_param: 'np.ndarray \| None' = None, grad_fn: 'Optional[Callable]' = None) -> 'Dict[str, Any]'` |
| `step` | `(self, x: 'np.ndarray', lr: 'float', grad: 'np.ndarray', state: 'Dict[str, Any]')` |

## `kfc_procedure.cobra.core.optimizers.gradient.base`

### `BaseGradientOptimizer`

```python
BaseGradientOptimizer(self, learning_rate: 'float' = 0.01, max_iter: 'int' = 300, tol: 'float' = 1e-07, speed: 'str' = 'constant', gradient_method: 'str' = 'central', eps: 'float' = 1e-07, n_tries: 'int' = 5, init_range=(0.0001, 3.0), show_process: 'bool' = True, **kwargs)
```

Base class for gradient-based optimization algorithms.

#### Methods

| Method | Signature |
|---|---|
| `gradient` | `(self, objective: 'Callable', params: 'np.ndarray', grad_fn: 'Optional[Callable]' = None) -> 'np.ndarray'` |
| `optimize` | `(self, objective: 'Callable[[np.ndarray], float]', init_param: 'np.ndarray \| None' = None, grad_fn: 'Optional[Callable]' = None) -> 'Dict[str, Any]'` |
| `step` | `(self, x: 'np.ndarray', lr: 'float', grad: 'np.ndarray', state: 'Dict[str, Any]')` |

## `kfc_procedure.cobra.core.optimizers.gradient.gd`

### `GradientDescentOptimizer`

```python
GradientDescentOptimizer(self, learning_rate: 'float' = 0.01, max_iter: 'int' = 300, tol: 'float' = 1e-07, speed: 'str' = 'constant', gradient_method: 'str' = 'central', eps: 'float' = 1e-07, n_tries: 'int' = 5, init_range=(0.0001, 3.0), show_process: 'bool' = True, **kwargs)
```

Standard Gradient Descent optimizer.

#### Methods

| Method | Signature |
|---|---|
| `gradient` | `(self, objective: 'Callable', params: 'np.ndarray', grad_fn: 'Optional[Callable]' = None) -> 'np.ndarray'` |
| `optimize` | `(self, objective: 'Callable[[np.ndarray], float]', init_param: 'np.ndarray \| None' = None, grad_fn: 'Optional[Callable]' = None) -> 'Dict[str, Any]'` |
| `step` | `(self, x: 'np.ndarray', lr: 'float', grad: 'np.ndarray', state: 'Dict[str, Any]')` |

## `kfc_procedure.cobra.core.optimizers.gradient.momentum`

### `MomentumOptimizer`

```python
MomentumOptimizer(self, momentum: 'float' = 0.9, **kwargs)
```

Momentum-based gradient descent optimizer.

#### Methods

| Method | Signature |
|---|---|
| `gradient` | `(self, objective: 'Callable', params: 'np.ndarray', grad_fn: 'Optional[Callable]' = None) -> 'np.ndarray'` |
| `optimize` | `(self, objective: 'Callable[[np.ndarray], float]', init_param: 'np.ndarray \| None' = None, grad_fn: 'Optional[Callable]' = None) -> 'Dict[str, Any]'` |
| `step` | `(self, x: 'np.ndarray', lr: 'float', grad: 'np.ndarray', state: 'Dict[str, Any]')` |

## `kfc_procedure.cobra.core.optimizers.search.base`

### `BaseSearchOptimizer`

```python
BaseSearchOptimizer(self, show_process: 'bool' = True, risk_strategy: 'str' = 'mean', **kwargs)
```

Base class for derivative-free (search-based) optimizers.

#### Methods

| Method | Signature |
|---|---|
| `candidates` | `(self) -> 'np.ndarray'` |
| `optimize` | `(self, objective: 'Callable[[np.ndarray], Any]', init_param: 'np.ndarray \| None' = None) -> 'Dict[str, Any]'` |
| `reduce_risk` | `(self, score)` |
| `select_best_index` | `(self, risks: 'np.ndarray') -> 'int'` |

## `kfc_procedure.cobra.core.optimizers.search.search`

### `GridSearchOptimizer`

```python
GridSearchOptimizer(self, param_grid: 'Dict[str, List[float]]', **kwargs)
```

Grid Search optimizer.

#### Methods

| Method | Signature |
|---|---|
| `candidates` | `(self) -> 'np.ndarray'` |
| `optimize` | `(self, objective: 'Callable[[np.ndarray], Any]', init_param: 'np.ndarray \| None' = None) -> 'Dict[str, Any]'` |
| `reduce_risk` | `(self, score)` |
| `select_best_index` | `(self, risks: 'np.ndarray') -> 'int'` |

## `kfc_procedure.cobra.core.splitters.base`

### `BaseDataSplitter`

```python
BaseDataSplitter(self, /, *args, **kwargs)
```

Abstract interface for dataset splitting strategies.

#### Methods

| Method | Signature |
|---|---|
| `split` | `(self, x: 'np.ndarray', y: 'np.ndarray', *, groups: 'np.ndarray \| None' = None) -> 'SplitIndices'` |

### `SplitterFactory`

```python
SplitterFactory(self, /, *args, **kwargs)
```

Factory for splitter registration and creation.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.cobra.core.splitters.holdout`

### `RandomHoldoutSplitter`

```python
RandomHoldoutSplitter(self, calibration_size: 'float' = 0.5, random_state: 'int | None' = None) -> 'None'
```

Random holdout data splitter.

#### Methods

| Method | Signature |
|---|---|
| `split` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'SplitIndices'` |

## `kfc_procedure.cobra.core.splitters.overlap`

### `OverlapSplitter`

```python
OverlapSplitter(self, split_ratio: 'float' = 0.5, overlap: 'float' = 0.0, shuffle: 'bool' = True, random_state: 'int | None' = None) -> 'None'
```

Splitter supporting overlapping partitions.

#### Methods

| Method | Signature |
|---|---|
| `split` | `(self, x: 'np.ndarray', y: 'np.ndarray') -> 'SplitIndices'` |

## `kfc_procedure.cobra.core.types`

### `SplitIndices`

```python
SplitIndices(self, train_idx: 'np.ndarray', eval_idx: 'np.ndarray', fold_id: 'int | None' = None) -> None
```

Index sets defining a train/evaluation partition.

### `TrainingContext`

```python
TrainingContext(self, X_k: 'np.ndarray | None', y_k: 'np.ndarray | None', X_l: 'np.ndarray', y_l: 'np.ndarray', as_predictions: 'bool') -> None
```

Container holding the datasets required during model training.

## `kfc_procedure.cobra.gradientcobra`

### `GradientCOBRA`

```python
GradientCOBRA(self, estimators: 'List[Union[str, BaseEstimator]] | None' = None, estimators_params: 'dict[str, Any] | None' = None, distance: 'str' = 'euclidean', distance_params: 'dict[str, Any] | None' = None, kernel: 'str' = 'rbf', kernel_params: 'dict[str, Any] | None' = None, aggregator: 'str' = 'weighted_mean', aggregator_params: 'dict[str, Any] | None' = None, loss: 'str' = 'mse', loss_params: 'dict[str, Any] | None' = None, optimizer: 'str' = 'grid', optimizer_params: 'dict[str, Any] | None' = None, opt_method: 'str' = 'grid', bandwidth_list: 'np.ndarray | None' = None, learning_rate: 'float' = 0.1, max_iter: 'int' = 300, n_cv: 'int' = 5, norm_constant: 'float | None' = None, n_jobs: 'int' = -1, random_state: 'int | None' = None)
```

Base class for all estimators in scikit-learn.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X, y, X_l=None, y_l=None, split_ratio=0.5, overlap=0.0, as_predictions=False)` |
| `get_params` | `(self, deep=True)` |
| `kappa_cross_validation_error` | `(self, params)` |
| `predict` | `(self, X)` |
| `score` | `(self, X, y, sample_weight=None)` |
| `set_params` | `(self, **params)` |
| `set_score_request` | `(self: kfc_procedure.cobra.gradientcobra.GradientCOBRA, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> kfc_procedure.cobra.gradientcobra.GradientCOBRA` |

## `kfc_procedure.cobra.mixcobra`

### `MixCOBRARegressor`

```python
MixCOBRARegressor(self, estimators: 'list[str | BaseEstimator] | None' = None, estimators_params: 'dict[str, Any] | None' = None, distance: 'str' = 'euclidean', distance_params: 'dict[str, Any] | None' = None, kernel: 'str' = 'rbf', kernel_params: 'dict[str, Any] | None' = None, aggregator: 'str' = 'weighted_mean', aggregator_params: 'dict[str, Any] | None' = None, loss: 'str' = 'mse', loss_params: 'dict[str, Any] | None' = None, optimizer: 'str' = 'grid', optimizer_params: 'dict[str, Any] | None' = None, norm_constant_x=None, norm_constant_y=None, alpha_list: 'np.ndarray | None' = None, beta_list: 'np.ndarray | None' = None, opt_method: 'str' = 'grid', learning_rate: 'float' = 0.01, max_iter: 'int' = 300, n_cv: 'int' = 5, speed: 'str' = 'constant', n_jobs: 'int' = 1, one_parameter: 'bool' = False, random_state: 'int | None' = None)
```

MixCOBRA regressor that learns mixing weights across input/output spaces.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray', X_l: 'np.ndarray \| None' = None, y_l: 'np.ndarray \| None' = None, split_ratio: 'float' = 0.5, overlap: 'float' = 0.0, pred_features: 'np.ndarray \| None' = None, as_predictions=False)` |
| `get_params` | `(self, deep=True)` |
| `kappa_cross_validation_error_1d` | `(self, params)` |
| `kappa_cross_validation_error_2d` | `(self, params)` |
| `predict` | `(self, X: 'np.ndarray', pred_X: 'np.ndarray \| None' = None, alpha: 'float \| None' = None, beta: 'float \| None' = None, bandwidth: 'float \| None' = None) -> 'np.ndarray'` |
| `score` | `(self, X, y, sample_weight=None)` |
| `set_params` | `(self, **params)` |
| `set_score_request` | `(self: kfc_procedure.cobra.mixcobra.MixCOBRARegressor, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> kfc_procedure.cobra.mixcobra.MixCOBRARegressor` |

## `kfc_procedure.cobra.superlearner`

### `SuperLearner`

```python
SuperLearner(self, random_state=None, base_learners=None, base_params=None, meta_learners=None, meta_params_cv=None, n_fold=10, cv_folds=None, loss_function=None, loss_weight=None)
```

Base class for all estimators in scikit-learn.

#### Methods

| Method | Signature |
|---|---|
| `add_extra_learners` | `(self, extra_learner)` |
| `draw_learning_curve` | `(self, y_test=None, fig_type='qq', save_fig=False, fig_path=False, show_fig=True)` |
| `fit` | `(self, X, y, train_meta_learners=True, as_predictions=False, show_warning=True)` |
| `get_params` | `(self, deep=True)` |
| `loss_func` | `(self, y_true, pred)` |
| `mae` | `(self, y_true, pred)` |
| `mape` | `(self, y_true, pred)` |
| `mse` | `(self, y_true, pred)` |
| `predict` | `(self, X, extra_features=None)` |
| `set_params` | `(self, **params)` |
| `train_base_learners` | `(self, final=False)` |
| `train_meta_learners` | `(self)` |

## `kfc_procedure.cobra.utils.preprocessing`

### Functions

| Function | Signature |
|---|---|
| `clean_sklearn_name` | `(name: 'str') -> 'str'` |
| `compute_normalization_constant` | `(y: 'np.ndarray', norm_constant: 'float \| None' = None, scale_factor: 'float' = 30.0, M: 'int' = 1) -> 'float'` |
| `data_split_overlap` | `(X: 'np.ndarray', y: 'np.ndarray', split: 'float' = 0.5, overlap: 'float' = 0.0, shuffle: 'bool' = True, random_state: 'int' = None) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]'` |
| `history_to_dataframe` | `(history, param_names=None)` |

## `kfc_procedure.cobra.utils.resolve`

### Functions

| Function | Signature |
|---|---|
| `fit_estimators` | `(X, y, estimators_params=None, estimators=None, n_jobs=1)` |
| `predict_estimators` | `(X: 'np.ndarray', estimators, n_jobs: 'int' = 1)` |
| `resolve_from_aggregator` | `(aggregator: 'str \| Any', aggregator_params: 'dict[str, Any] \| None')` |
| `resolve_from_distance` | `(distance: 'str \| Any', distance_params: 'dict[str, Any] \| None')` |
| `resolve_from_estimators` | `(estimators: 'list[Any] \| str \| None', estimators_params: 'dict[str, Any] \| None', default_estimators: 'list[str]') -> 'list[Any]'` |
| `resolve_from_kernel` | `(kernel: 'str \| Any', kernel_params: 'dict[str, Any] \| None')` |
| `resolve_from_loss` | `(loss: 'str \| Any', loss_params: 'dict[str, Any] \| None')` |
| `resolve_from_splitter` | `(splitter: 'str \| Any', splitter_params: 'dict[str, Any] \| None')` |
| `resolve_training_context` | `(X: 'ArrayLike', y: 'ArrayLike', *, X_l: 'ArrayLike \| None' = None, y_l: 'ArrayLike \| None' = None, pred_features: 'ArrayLike \| None' = None, as_predictions: 'bool' = False, splitter: 'BaseDataSplitter \| None' = None, split_ratio: 'float' = 0.5, overlap: 'float' = 0.0, random_state: 'int \| None' = None) -> 'TrainingContext'` |

## `kfc_procedure.core.clustering.bregman`

### `BregmanKMeans`

```python
BregmanKMeans(self, n_clusters: 'int' = 8, *, divergence: 'BaseBregmanDivergence', n_init: 'int' = 10, max_iter: 'int' = 300, tol: 'float' = 0.0001, random_state=None, verbose: 'bool' = False)
```

K-Means clustering with Bregman divergences.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'ArrayLike', y=None, init=None)` |
| `fit_predict` | `(self, X, y=None)` |
| `fit_transform` | `(self, X, y=None)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'ArrayLike') -> 'NDArray'` |
| `score` | `(self, X, y=None) -> 'float'` |
| `set_output` | `(self, *, transform=None)` |
| `set_params` | `(self, **params)` |
| `transform` | `(self, X: 'ArrayLike') -> 'NDArray'` |

### Functions

| Function | Signature |
|---|---|
| `validate_divergence_domain` | `(div: 'BaseBregmanDivergence', X: 'np.ndarray') -> 'None'` |

## `kfc_procedure.core.clustering.divergences.base`

### `BaseBregmanDivergence`

```python
BaseBregmanDivergence(self, validate_domain: 'bool' = True, **kwargs)
```

Abstract base class for Bregman divergences.

#### Methods

| Method | Signature |
|---|---|
| `assign_clusters` | `(self, X: 'np.ndarray', centroids: 'np.ndarray') -> 'np.ndarray'` |
| `centroid` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `distance` | `(self, X: 'np.ndarray', Y: 'np.ndarray', *, clip: 'bool' = True) -> 'np.ndarray'` |
| `grad_phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `in_domain` | `(self, X: 'np.ndarray') -> 'bool'` |
| `pairwise` | `(self, X: 'np.ndarray', Y: 'np.ndarray') -> 'np.ndarray'` |
| `phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |

### `BregmanDivergenceFactory`

```python
BregmanDivergenceFactory(self, /, *args, **kwargs)
```

Factory class for Bregman divergence implementations.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.core.clustering.divergences.euclidean`

### `SquaredEuclidean`

```python
SquaredEuclidean(self, validate_domain: 'bool' = True, **kwargs)
```

Squared Euclidean distance.

#### Methods

| Method | Signature |
|---|---|
| `assign_clusters` | `(self, X: 'np.ndarray', centroids: 'np.ndarray') -> 'np.ndarray'` |
| `centroid` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `distance` | `(self, X: 'np.ndarray', Y: 'np.ndarray', *, clip: 'bool' = True) -> 'np.ndarray'` |
| `grad_phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `in_domain` | `(self, X: 'np.ndarray') -> 'bool'` |
| `pairwise` | `(self, X: 'np.ndarray', Y: 'np.ndarray') -> 'np.ndarray'` |
| `phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |

## `kfc_procedure.core.clustering.divergences.gkl`

### `GKLDivergence`

```python
GKLDivergence(self, validate_domain: 'bool' = True, **kwargs)
```

Generalised Kullback-Leibler (I-divergence).

#### Methods

| Method | Signature |
|---|---|
| `assign_clusters` | `(self, X: 'np.ndarray', centroids: 'np.ndarray') -> 'np.ndarray'` |
| `centroid` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `distance` | `(self, X: 'np.ndarray', Y: 'np.ndarray', *, clip: 'bool' = True) -> 'np.ndarray'` |
| `grad_phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `in_domain` | `(self, X: 'np.ndarray') -> 'bool'` |
| `pairwise` | `(self, X: 'np.ndarray', Y: 'np.ndarray') -> 'np.ndarray'` |
| `phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |

## `kfc_procedure.core.clustering.divergences.itakura_saito`

### `ItakuraSaito`

```python
ItakuraSaito(self, validate_domain: 'bool' = True, **kwargs)
```

Itakura-Saito divergence.

#### Methods

| Method | Signature |
|---|---|
| `assign_clusters` | `(self, X: 'np.ndarray', centroids: 'np.ndarray') -> 'np.ndarray'` |
| `centroid` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `distance` | `(self, X: 'np.ndarray', Y: 'np.ndarray', *, clip: 'bool' = True) -> 'np.ndarray'` |
| `grad_phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `in_domain` | `(self, X: 'np.ndarray') -> 'bool'` |
| `pairwise` | `(self, X: 'np.ndarray', Y: 'np.ndarray') -> 'np.ndarray'` |
| `phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |

## `kfc_procedure.core.clustering.divergences.logistic`

### `LogisticLoss`

```python
LogisticLoss(self, validate_domain: 'bool' = True, **kwargs)
```

Logistic / binary cross-entropy loss.

#### Methods

| Method | Signature |
|---|---|
| `assign_clusters` | `(self, X: 'np.ndarray', centroids: 'np.ndarray') -> 'np.ndarray'` |
| `centroid` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `distance` | `(self, X: 'np.ndarray', Y: 'np.ndarray', *, clip: 'bool' = True) -> 'np.ndarray'` |
| `grad_phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `in_domain` | `(self, X: 'np.ndarray') -> 'bool'` |
| `pairwise` | `(self, X: 'np.ndarray', Y: 'np.ndarray') -> 'np.ndarray'` |
| `phi` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |

## `kfc_procedure.core.combiner.base`

### `BaseCombiner`

```python
BaseCombiner(self, /, *args, **kwargs)
```

Abstract base class for ensemble combination strategies.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `fit` | `(self, X: 'np.ndarray', y: 'Optional[np.ndarray]' = None)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

### `CombinerFactory`

```python
CombinerFactory(self, /, *args, **kwargs)
```

Abstract base class for registry-driven factories.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.core.combiner.classification.combined_classifier`

### `CobraClassifierCombiner`

```python
CobraClassifierCombiner(self, **cobra_params)
```

COBRA-based classifier combiner.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y: numpy.ndarray)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.classification.majority_vote`

### `MajorityVoteCombiner`

```python
MajorityVoteCombiner(self, /, *args, **kwargs)
```

Hard voting ensemble combiner.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y=None)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.classification.stacking`

### `StackingClassifierCombiner`

```python
StackingClassifierCombiner(self, meta_model=None)
```

Logistic regression stacking classifier.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y: numpy.ndarray)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.regression.gradientcobra`

### `GradientCOBRACombiner`

```python
GradientCOBRACombiner(self, **cobra_params)
```

GradientCOBRA-based regression combiner.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y: numpy.ndarray)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.regression.mean`

### `MeanCombiner`

```python
MeanCombiner(self, /, *args, **kwargs)
```

Row-wise mean combiner for regression ensembles.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y=None)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.regression.mixcobra`

### `MixCOBRACombiner`

```python
MixCOBRACombiner(self, **cobra_params)
```

MixCOBRA-based regression combiner.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y: numpy.ndarray)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.regression.stacking`

### `StackingRegressorCombiner`

```python
StackingRegressorCombiner(self, meta_model=None)
```

Stacking combiner using a regression meta-model.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y: numpy.ndarray)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.combiner.regression.weighted_mean`

### `WeightedMeanCombiner`

```python
WeightedMeanCombiner(self, fit_intercept: bool = False)
```

Linear regression-based weighted combiner.

#### Methods

| Method | Signature |
|---|---|
| `combine` | `(self, X: numpy.ndarray) -> numpy.ndarray` |
| `fit` | `(self, X: numpy.ndarray, y: numpy.ndarray)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.factory`

### `BaseFactory`

```python
BaseFactory(self, /, *args, **kwargs)
```

Abstract base class for registry-driven factories.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.core.ml.base`

### `BaseLocalModel`

```python
BaseLocalModel(self, /, *args, **kwargs)
```

Unified base class for all local models in the F-step.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

### `LocalModelFactory`

```python
LocalModelFactory(self, /, *args, **kwargs)
```

Factory for all local models used in F-step.

#### Methods

| Method | Signature |
|---|---|
| `available` | `() -> 'List[str]'` |
| `available_by_category` | `(category: 'str') -> 'List[str]'` |
| `available_categories` | `() -> 'Set[str]'` |
| `contains` | `(name: 'str') -> 'bool'` |
| `create` | `(name: 'str', **kwargs) -> 'Any'` |
| `find_by_class` | `(target_cls: 'Type') -> 'List[str]'` |
| `info` | `(name: 'str') -> 'Dict[str, Any]'` |
| `register` | `(*names: 'str', categories: 'Optional[Set[str] \| str]' = None, **metadata: 'Any')` |
| `supports` | `(name: 'str', category: 'str') -> 'bool'` |

## `kfc_procedure.core.ml.sklearn`

### `MeanRegressor`

```python
MeanRegressor(self) -> 'None'
```

Unified base class for all local models in the F-step.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'ArrayLike', y: 'ArrayLike')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'ArrayLike') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

### `SklearnLocalModel`

```python
SklearnLocalModel(self, model_cls: 'Type[BaseEstimator]', **kwargs)
```

Adapter wrapping sklearn estimators into F-step local models.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X, y)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X)` |
| `predict_proba` | `(self, X)` |
| `set_params` | `(self, **params)` |

### Functions

| Function | Signature |
|---|---|
| `clean_sklearn_name` | `(name: 'str') -> 'str'` |
| `register_all_sklearn_models` | `()` |

## `kfc_procedure.core.steps.cstep`

### `CStep`

```python
CStep(self, combiner: 'Union[str, BaseCombiner]', combiner_params: 'Optional[Dict]' = None, task: 'str' = 'regression', random_state: 'Optional[int]' = None)
```

C-step: Aggregation layer for divergence-aware predictions.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.steps.fstep`

### `FStep`

```python
FStep(self, local_model: 'Union[str, BaseLocalModel]', local_model_params: 'Optional[Dict]' = None, task: 'str' = 'regression', random_state: 'Optional[int]' = None)
```

Local model fitting stage of the KFC pipeline.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X, y, clusters: 'Dict[str, np.ndarray]')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X, clusters: 'Dict[str, np.ndarray]')` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.core.steps.kstep`

### `KStep`

```python
KStep(self, divergences: 'List[Union[str, BaseBregmanDivergence]]', divergences_params: 'Dict' = {}, n_clusters: 'int' = 3, max_iter: 'int' = 300, tol: 'float' = 0.0001, verbose: 'bool' = False, random_state: 'int | None' = None)
```

Multi-divergence clustering stage in the KFC pipeline.

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray \| None' = None)` |
| `fit_predict` | `(self, X, y=None, **kwargs)` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'Dict[str, np.ndarray]'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.kfc`

### `KFCClassifier`

```python
KFCClassifier(self, *args, **kwargs)
```

Full KFC pipeline estimator (sklearn-compatible).

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

### `KFCProcedure`

```python
KFCProcedure(self, divergences, local_model, combiner, divergences_params: 'Optional[Dict]' = None, local_model_params: 'Optional[Dict]' = None, combiner_params: 'Optional[Dict]' = None, task: 'str' = 'regression', n_clusters=3, max_iter=300, tol=0.0001, verbose=0, random_state=None)
```

Full KFC pipeline estimator (sklearn-compatible).

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

### `KFCRegressor`

```python
KFCRegressor(self, *args, **kwargs)
```

Full KFC pipeline estimator (sklearn-compatible).

#### Methods

| Method | Signature |
|---|---|
| `fit` | `(self, X: 'np.ndarray', y: 'np.ndarray')` |
| `get_params` | `(self, deep=True)` |
| `predict` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `predict_proba` | `(self, X: 'np.ndarray') -> 'np.ndarray'` |
| `set_params` | `(self, **params)` |

## `kfc_procedure.utils.logger`

### `Logger`

```python
Logger(self, verbose: int = 0)
```

Lightweight structured logger for the KFC pipeline.

#### Methods

| Method | Signature |
|---|---|
| `debug` | `(self, msg: str)` |
| `info` | `(self, msg: str)` |
| `log` | `(self, level: int, msg: str)` |
| `trace` | `(self, msg: str)` |

### Functions

| Function | Signature |
|---|---|
| `timed` | `(logger, name)` |

## `kfc_procedure.utils.resolve`

### Functions

| Function | Signature |
|---|---|
| `resolve_bregman` | `(cfg: Union[str, Dict[str, Any], kfc_procedure.core.clustering.bregman.BregmanKMeans]) -> kfc_procedure.core.clustering.bregman.BregmanKMeans` |
| `resolve_kstep` | `(cfgs: List[Union[str, Dict[str, Any], kfc_procedure.core.clustering.bregman.BregmanKMeans]]) -> Dict[str, kfc_procedure.core.clustering.bregman.BregmanKMeans]` |
