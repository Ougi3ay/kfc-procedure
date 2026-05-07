from __future__ import annotations

from typing import Any, List, Union

import numpy as np

from joblib import Parallel, delayed

from sklearn.base import (
    BaseEstimator as SkBaseEstimator,
    RegressorMixin,
)
from sklearn.model_selection import KFold
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)

from cobra.core.adapters.base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)
from cobra.core.aggregators.base import (
    BaseAggregator,
    AggregatorFactory,
)
from cobra.core.distances.base import (
    BaseDistance,
    DistanceFactory,
)
from cobra.core.estimators.base import BaseEstimator
from cobra.core.kernels.base import (
    BaseKernel,
    KernelFactory,
)
from cobra.core.losses.base import (
    BaseLoss,
    LossFactory,
)
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.spaces.base import (
    BaseSpaceNormalizer,
    SpaceNormalizerFactory,
)
from cobra.core.splitters.base import (
    BaseDataSplitter,
    SplitterFactory,
)

from cobra.utils.resolve import fit_estimators_parallel


class GradientCOBRA(SkBaseEstimator, RegressorMixin):
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
        optimizer: str = "grid",
        optimizer_params: dict[str, Any] | None = None,
        opt_method: str = "grid",
        bandwidth_list: np.ndarray | None = None,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        n_cv: int = 5,
        split_ratio: float = 0.5,
        overlap: float = 0.0,
        norm_constant: float | None = None,
        n_jobs: int = -1,
        random_state: int | None = None,
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
        self.opt_method = opt_method
        self.bandwidth_list = bandwidth_list
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_cv = n_cv
        self.split_ratio = split_ratio
        self.overlap = overlap
        self.norm_constant = norm_constant
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def _resolve_fit_split_context(
        self,
        X,
        y,
        X_l=None,
        y_l=None,
        as_predictions=False,
    ):
        """
        Return : X_k, y_k, X_l, y_l
        """
        X, y = check_X_y(X, y)
        if as_predictions:
            self.as_predictions_ = True
            return None, None, X, y
        if X_l is not None and y_l is not None:
            self.as_predictions_ = False
            X_l, y_l = check_X_y(X_l, y_l)
            return X, y, X_l, y_l
        
        self.as_predictions_ = False
        splitter: BaseDataSplitter = SplitterFactory.create(
            "split_overlap",
            split_ratio=self.split_ratio,
            overlap=self.overlap,
            random_state=self.random_state,
        )
        iloc_k, iloc_l = splitter.split(X, y)
        return X[iloc_k], y[iloc_k], X[iloc_l], y[iloc_l]
    
    def _fit_estimators(self, X_k, y_k):
        default_estimators = [
            "linear_regression",
            "ridge_cv",
            "lasso_cv",
            "k_neighbors_regressor",
            "random_forest_regressor",
            "svr",
        ]

        estimators = self.estimators or default_estimators

        return fit_estimators_parallel(
            X=X_k,
            y=y_k,
            estimators=estimators,
            estimators_params=self.estimators_params,
            n_jobs=self.n_jobs,
        )
    
    def _load_predictions(self, X):
        def _predict_single(estimator):
            return estimator.predict(X)
        preds = Parallel(
            n_jobs=self.n_jobs,
            verbose=0,
        )(
            delayed(_predict_single)(est)
            for est in self.estimators_
        )
        return np.column_stack(preds)
    
    def _space_normalize(
        self,
        X,
        prediction_space,
    ):
        normalizer: BaseSpaceNormalizer = SpaceNormalizerFactory.create(
            "gradientcobra",
            norm_constant=self.norm_constant,
        )
        return normalizer.transform(
            X,
            prediction_space,
        )
    
    def _resolve_components(self):
        self.distance_: BaseDistance = (
            DistanceFactory.create(
                self.distance,
                **(self.distance_params or {}),
            )
        )

        self.kernel_: BaseKernel = (
            KernelFactory.create(
                self.kernel,
                **(self.kernel_params or {}),
            )
        )

        self.aggregator_: BaseAggregator = (
            AggregatorFactory.create(
                self.aggregator,
                **(self.aggregator_params or {}),
            )
        )

        self.loss_: BaseLoss = (
            LossFactory.create(
                self.loss,
                **(self.loss_params or {}),
            )
        )

        self.adapter_: BaseKernelAdapter = (
            KernelAdapterFactory.create(
                "gradientcobra",
                bandwidth=1.0,
            )
        )

        self.splitter_ : BaseDataSplitter = (
            SplitterFactory.create(
                "kfold",
                n_splits=self.n_cv,
                shuffle=True,
                random_state=self.random_state,
            )
        )
    
    def kappa_cross_validation_error(self, params):
        bandwidth = float(
            np.atleast_1d(params)[0]
        )

        self.adapter_.set_params(bandwidth=bandwidth)
        D = self.adapter_.transform(self.distance_matrix_)
        K = self.kernel_(D)
        preds = np.empty(len(self.y_l_), dtype=float)

        for train_idx, val_idx in self.cv_folds_:
            K_val_train = K[np.ix_(val_idx, train_idx)]
            y_train = self.y_l_[train_idx]
            numerator = K_val_train @ y_train
            denominator = np.sum(K_val_train, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                pred_fold = np.where(
                    denominator > 0,
                    numerator / denominator,
                    np.mean(y_train),
                )
            preds[val_idx] = pred_fold
        
        return self.loss_(self.y_l_, preds)

    def _optimize_hyperparameters(self):
        if self.bandwidth_list is None:
            bandwidth_candidates = np.linspace(
                0.001,
                10.0,
                self.max_iter,
            )
        else:
            bandwidth_candidates = self.bandwidth_list
        
        if self.opt_method == "grid":
            self.optimizer_ = OptimizerFactory.create(
                self.optimizer,
                param_grid={
                    "bandwidth": bandwidth_candidates,
                },
                random_state=self.random_state,
                **(self.optimizer_params or {}),
            )
            result = self.optimizer_(self.kappa_cross_validation_error)
        
        elif self.opt_method == "grad":
            if self.optimizer == "grid":
                self.optimizer = "grad"
            
            self.optimizer_ = OptimizerFactory.create(
                self.optimizer,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                **(self.optimizer_params or {}),
            )
            result = self.optimizer_(self.kappa_cross_validation_error)
        
        else:
            raise ValueError(
                f"Unknown opt_method={self.opt_method}"
            )
        
        self.bandwidth_ = float(np.atleast_1d(result["x"])[0])
        
        self.optimization_outputs_ = {
            "method": self.opt_method,
            "bandwidth": self.bandwidth_,
            "score": result["score"],
            "history": result["history"],
            "evaluations": len(result["history"]),
        }
    
    def fit(
        self,
        X,
        y,
        X_l=None,
        y_l=None,
        as_predictions=False,
    ):
        (
            self.X_k_,
            self.y_k_,
            self.X_l_,
            self.y_l_,
        ) = self._resolve_fit_split_context(
            X,
            y,
            X_l,
            y_l,
            as_predictions,
        )

        if not self.as_predictions_:
            self.estimators_ = self._fit_estimators(
                self.X_k_,
                self.y_k_,
            )

            prediction_space = self._load_predictions(
                self.X_l_,
            )
        else:
            prediction_space = self.X_l_
        
        (
            self.X_l_norm_,
            self.Y_l_norm_,
        ) = self._space_normalize(
            self.X_l_,
            prediction_space,
        )

        self._resolve_components()

        self.distance_matrix_ = self.distance_.matrix(
            self.Y_l_norm_,
            self.Y_l_norm_
        )

        self.cv_folds_ = self.splitter_.split(self.X_l_norm_, self.Y_l_norm_)

        self._optimize_hyperparameters()

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.as_predictions_:
            prediction_space = X
        else:
            prediction_space = self._load_predictions(X)
        
        (
            X_norm,
            Y_norm,
        ) = self._space_normalize(
            X,
            prediction_space,
        )

        distance_matrix = self.distance_.matrix(
            Y_norm,
            self.Y_l_norm_,
        )

        self.adapter_.set_params(bandwidth=self.bandwidth_)

        D = self.adapter_.transform(distance_matrix)
        K = self.kernel_(D)

        preds = np.empty(len(X), dtype=float)

        for i in range(len(X)):
            weights = K[i]
            denominator = np.sum(
                weights
            )

            if denominator > 0:
                preds[i] = self.aggregator_.aggregate(self.y_l_, weights)
            else:
                preds[i] = np.mean(self.y_l_)
        
        return preds

