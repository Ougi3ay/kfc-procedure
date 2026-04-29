"""
GradientCOBRA

A gradient-optimized extension of the COBRA framework for consensus-based
regression using kernel-weighted aggregation in a learned prediction space.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Overview
--------
GradientCOBRA extends classical COBRA by introducing:

- differentiable/optimizable kernel bandwidth
- gradient-based or search-based hyperparameter tuning
- normalized prediction space alignment
- cross-validated loss-driven calibration

Core idea
---------
Instead of fixed kernel bandwidth, GradientCOBRA learns optimal
smoothing parameters by minimizing a validation loss over:

- estimator prediction space
- kernel-induced similarity graph
- aggregation-based reconstruction error

This enables adaptive consensus formation.

Design goals
------------
- optimize kernel behavior (bandwidth tuning)
- support gradient + grid-based optimization
- unify estimator + kernel + loss pipeline
- support direct (X_l, y_l) or internal split mode
- maintain sklearn-compatible API

Key components
--------------
- Estimators: base prediction models
- Distance: similarity in prediction space
- Kernel: converts distances → weights
- Kernel Adapter: parameterized transformation layer
- Aggregator: weighted consensus function
- Loss: optimization objective
- Optimizer: gradient or search-based tuning
- Splitter: train/aggregation partitioning
- Space Normalizer: stabilizes representation scale

Optimization modes
-------------------
1. Gradient-based (`opt_method="grad"`)
   - numerical gradient descent
   - continuous bandwidth tuning

2. Search-based (`opt_method="search"`)
   - grid or random search
   - discrete parameter evaluation

"""
from __future__ import annotations

from abc import ABC
from typing import Any, List, Union
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator as SkBaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted

from cobra.core.adapters.base import BaseKernelAdapter, KernelAdapterFactory
from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel, KernelFactory
from cobra.core.losses.base import BaseLoss, LossFactory
from cobra.core.optimizers.gradient.base import GradientOptimizerFactory
from cobra.core.optimizers.search.base import SearchOptimizerFactory
from cobra.core.spaces.base import SpaceNormalizerFactory
from cobra.core.splitters.base import BaseDataSplitter, SplitterFactory

class GradientCOBRA(ABC, SkBaseEstimator, RegressorMixin):
    """
    GradientCOBRA regressor with differentiable kernel parameter tuning.

    GradientCOBRA extends the COBRA ensemble by enabling continuous
    optimization of kernel parameters (for example, bandwidth) using a
    differentiable objective. It supports both gradient-based optimizers and
    discrete search strategies for hyperparameter selection.

    Parameters
    ----------
    estimators : list[str | BaseEstimator] | None, default=None
        Identifiers or estimator instances forming the expert pool. String
        identifiers are resolved via ``EstimatorFactory``.

    estimators_params : dict[str, Any] | None, default=None
        Mapping from estimator identifier to constructor keyword arguments.

    distance : str, default='euclidean'
        Distance metric identifier used to compute pairwise distances in the
        prediction space.

    distance_params : dict | None, default=None
        Parameters forwarded to the distance implementation.

    kernel : str, default='rbf'
        Kernel identifier used to convert transformed distances to weights.

    kernel_params : dict | None, default=None
        Parameters forwarded to the kernel implementation.

    aggregator : str, default='weighted_mean'
        Aggregation strategy identifier for combining weighted neighbor
        predictions.

    aggregator_params : dict | None, default=None
        Keyword arguments forwarded to the aggregator implementation.

    loss : str, default='mse'
        Loss identifier used as the optimization objective.

    loss_params : dict | None, default=None
        Keyword arguments forwarded to the loss implementation.

    optimizer : str, default='gradient_descent'
        Optimizer identifier used for gradient-based tuning when
        ``opt_method='grad'``.

    optimizer_params : dict | None, default=None
        Parameters forwarded to the optimizer implementation.

    opt_method : str, default='grad'
        High-level optimization mode. ``'grad'`` uses a gradient optimizer
        (continuous tuning); ``'search'`` uses a grid or random search over
        a discrete parameter set.

    bandwidth_list : np.ndarray | None
        Optional candidate bandwidths for discrete search optimizers.

    norm_constant : float | None
        Optional normalization constant used by the space normalizer.

    random_state : int | None
        Random seed used by splitters, optimizers and any stochastic
        components.

    Notes
    -----
    - The implementation follows a clear pipeline: estimator training,
      prediction-matrix construction, normalization, distance/kernel
      computation, optimizer-driven parameter selection, and aggregation.
    - New components should be registered via the appropriate factories to
      integrate with the pipeline seamlessly.

    See Also
    --------
    MixCOBRARegressor, CombineClassifier
    """

    def __init__(
        self,
        estimators: List[Union[str, BaseEstimator]] | None = None,
        estimators_params: dict[str, Any] | None = None,
        distance: str = "euclidean",
        distance_params: dict[str, Any] | None = None,
        kernel: str = "rbf",
        kernel_params: dict[str, Any] | None = None,
        aggregator: str = "weighted_mean",
        aggregator_params: dict[str, Any] | None = None,
        loss: str = "mse",
        loss_params: dict[str, Any] | None = None,
        optimizer: str = "gradient_descent",
        optimizer_params: dict[str, Any] | None = None,

        opt_method: str = "grad",
        bandwidth_list: np.ndarray | None = None,
        norm_constant = None,
        random_state: int | None = None
    ):
        self.estimators = estimators
        self.estimators_params = estimators_params
        self.distance = distance
        self.distance_params = distance_params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.aggregator = aggregator
        self.aggregator_params = aggregator_params
        self.loss = loss
        self.loss_params = loss_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        self.bandwidth_list = bandwidth_list
        self.norm_constant = norm_constant
        self.opt_method = opt_method
        self.random_state = random_state
    
    def _resolve_fit_split_context(self, X, y, X_l, y_l):
        """
        Build training and aggregation datasets.

        Returns
        -------
        tuple
            X_k, y_k : training set
            X_l, y_l : aggregation set
            iloc_k, iloc_l : indices
        """
        X, y = check_X_y(X, y)
        if X_l is not None and y_l is not None:
            X_l, y_l = check_X_y(X_l, y_l)
            X_k_, X_l_ = X, X_l
            y_k_, y_l_ = y, y_l
            iloc_l, iloc_k = np.arange(len(y_l_)), np.arange(len(y))
            self.as_predictions_ = True
        else:
            # provide static splitter
            splitter: BaseDataSplitter = SplitterFactory.create(
                "split_overlap",
                split_ratio=0.5,
                overlap=0.0,
                random_state=self.random_state
            )
            iloc_k, iloc_l = splitter.split(X, y)
            X_k_, y_k_ = X[iloc_k], y[iloc_k]
            X_l_, y_l_ = X[iloc_l], y[iloc_l]
            self.as_predictions_ = False

        return X_k_, y_k_, X_l_, y_l_, iloc_k, iloc_l
    
    def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):
        """
        Fit base estimator pool.
        """

        default_estimators = [
            "linear",
            "ridge",
            "lasso",
            "knn",
            "random_forest",
            "svm",
        ]

        estimators = self.estimators or default_estimators

        machines = []

        for est in estimators:
            if isinstance(est, str):
                params = (self.estimators_params or {}).get(est, {})
                model = EstimatorFactory.create(est, **params)
            elif isinstance(est, BaseEstimator):
                model = est
            else:
                raise ValueError(
                    f"Invalid estimator: {type(est)}. "
                    f"Expected str or BaseEstimator. "
                    f"Available: {EstimatorFactory.available()}"
                )

            model.fit(X_k, y_k)
            machines.append(model)
        
        return machines
    
    def _space_normalize(self, X, model_outputs):
        """
        Normalize estimator prediction space.
        """
        normalizer = SpaceNormalizerFactory.create(
            "gradientcobra",
            norm_constant=self.norm_constant
        )
        return normalizer.transform(X, model_outputs)
    
    def _load_predictions(self, X):
        """
        Build prediction matrix from estimator pool.
        """
        cols = []
        for model in self.estimators_:
            preds = model.predict(X)
            cols.append(preds)
        return np.column_stack(cols)
    
    def _resolve_component(self):
        """
        Initialize COBRA components (distance, kernel, aggregator, loss, adapter).
        """
        self.distance_ : BaseDistance = DistanceFactory.create(
            self.distance,
            **(self.distance_params or {})
        )

        self.kernel_ : BaseKernel = KernelFactory.create(
            self.kernel,
            **(self.kernel_params or {})
        )

        self.aggregator_ : BaseAggregator = AggregatorFactory.create(
            self.aggregator,
            **(self.aggregator_params or {})
        )

        self.loss_ : BaseLoss = LossFactory.create(
            self.loss,
            **(self.loss_params or {})
        )

        self.splitter_ : BaseDataSplitter = SplitterFactory.create(
            "kfold",
            n_splits=5,
            random_state=self.random_state
        )

        self.adapter_ : BaseKernelAdapter = KernelAdapterFactory.create(
            "gradientcobra",
            bandwidth=1.0
        )

    def _optimize_hyperparameters(self):
        """
        Optimize kernel bandwidth using selected strategy.

        Supports:
        - gradient descent
        - grid/search optimization
        """
        self.distance_matrix_ = self.distance_.matrix(self.Y_l_norm_, self.Y_l_norm_)

        folds = self.splitter_.split(self.X_l_, self.y_l_)

        def objective(params):
            # set bandwidth
            self.adapter_.set_params(bandwidth=params[0])

            D = self.adapter_.transform(self.distance_matrix_)
            K = self.kernel_(D)

            n_samples = len(self.y_l_)
            preds = np.empty(n_samples, dtype=float)

            for train_idx, val_idx in folds:
                w = K[train_idx][:, train_idx]
                y = self.y_l_[train_idx]

                denom = np.sum(w, axis=1)

                for i, v in enumerate(val_idx):
                    if denom[i] > 0:
                        preds[v] = self.aggregator_.aggregate(y, w[i])
                    else:
                        preds[v] = np.mean(y)
                
            return self.loss_(self.y_l_, preds)
        
        if self.opt_method == "grad":
            self.optimizer_ = GradientOptimizerFactory.create(
                self.optimizer,
                **(self.optimizer_params or {}),
                random_state=self.random_state
            )
            params, history = self.optimizer_(objective, np.array([1.0]))
        
        else:
            self.optimizer_ = SearchOptimizerFactory.create(
                self.optimizer,
                **(self.optimizer_params or {}),
                param_grid={"bandwidth" : self.bandwidth_list or np.linspace(0.1, 10.0, 20)},
                random_state=self.random_state
            )
            params, history = self.optimizer_(objective, self.bandwidth_list)

        self.optimization_outputs_ = {
            "method": self.opt_method,
            "params": params,
            "history": history
        }   


        
    def fit(
        self,
        X : np.ndarray,
        y : np.ndarray,
        X_l : np.ndarray | None = None,
        y_l : np.ndarray | None = None
    ):
        """
        Fit GradientCOBRA model with kernel bandwidth tuning.

        This method trains base estimators and learns the optimal kernel
        bandwidth (or other adapter parameters) that minimize prediction
        error on a calibration set.

        Parameters
        ----------
        X : np.ndarray
            Training features. Shape: (n_samples, n_features).

        y : np.ndarray
            Training targets. Shape: (n_samples,).

        X_l : np.ndarray | None, default=None
            External calibration features. If provided, used for aggregation
            instead of internal split. Shape: (n_cal_samples, n_features).

        y_l : np.ndarray | None, default=None
            External calibration targets. Shape: (n_cal_samples,).

        Returns
        -------
        self : GradientCOBRA
            Fitted model instance.

        Workflow
        --------
        1. Split data into training (X_k) and calibration (X_l) subsets
        2. Train base estimators on X_k
        3. Generate prediction matrix on X_l
        4. Normalize prediction space
        5. Initialize distance, kernel, adapter, and aggregator components
        6. Tune kernel bandwidth using optimization strategy

        Examples
        --------
        >>> model = GradientCOBRA()
        >>> model.fit(X_train, y_train)
        """
        # split data into train and aggregation sets
        (
            self.X_k_, self.y_k_,
            self.X_l_, self.y_l_,
            self.iloc_k_, self.iloc_l_
        ) = self._resolve_fit_split_context(X, y, X_l, y_l) 

        # fit base estimators on training set
        self.estimators_ = self._fit_estimators(self.X_k_, self.y_k_)

        # load predictions on aggregation set
        if not self.as_predictions_:
            model_outputs = self._load_predictions(self.X_l_)
        else:
            model_outputs = self.X_l_

        # normalize space
        self.X_l_norm_, self.Y_l_norm_ = self._space_normalize(self.X_l_, model_outputs)

        # resolve component
        self._resolve_component()

        self._optimize_hyperparameters()
        return self

    def predict(self, X):
        """
        Predict target values using fitted GradientCOBRA model.

        For each test sample, this method computes distances in the
        normalized prediction space, applies the optimized kernel with
        tuned bandwidth, and aggregates neighbor training targets.

        Parameters
        ----------
        X : np.ndarray
            Test features. Shape: (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted target values. Shape: (n_samples,).

        Workflow
        --------
        1. Generate predictions from all base estimators
        2. Normalize prediction space
        3. Compute distances to calibration predictions
        4. Transform distances using tuned kernel adapter
        5. Apply kernel function to generate similarity weights
        6. Aggregate calibration targets using kernel weights

        Examples
        --------
        >>> y_pred = model.predict(X_test)
        """
        check_is_fitted(self)

        model_outputs = self._load_predictions(X)
        X_norm, Y_norm = self._space_normalize(X, model_outputs)

        distance_matrix = self.distance_.matrix(Y_norm, self.Y_l_norm_)
        D = self.adapter_.transform(distance_matrix)
        K = self.kernel_(D)

        n_samples = len(X)
        preds = np.empty(n_samples, dtype=float)

        for i in range(n_samples):
            w = K[i]
            denom = np.sum(w)

            if denom > 0:
                preds[i] = self.aggregator_.aggregate(self.y_l_, w)
            else:
                preds[i] = np.mean(self.y_l_)
        
        return preds
