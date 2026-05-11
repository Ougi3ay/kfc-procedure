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
from cobra.core.validators.base import BaseCrossValidator, CVFactory
from cobra.utils.resolve import fit_estimators, resolve_training_context


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
        max_iter: int = 300,
        n_cv: int = 5,
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
        self.norm_constant = norm_constant
        self.n_jobs = n_jobs
        self.random_state = random_state
    
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

        return fit_estimators(
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

        self.cv_ : BaseCrossValidator = (
            CVFactory.create(
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
        errors = []

        for fold in self.cv_folds_:
            train_idx = fold.train_idx
            val_idx = fold.eval_idx

            # np.ix_ creates a 2D indexing grid (rows = validation, columns = training)
            K_val_train = K[np.ix_(val_idx, train_idx)]
            y_train = self.y_l_[train_idx]

            numerator = K_val_train @ y_train
            denominator = np.sum(K_val_train, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                pred_fold = np.where(
                    denominator > 0,
                    numerator / denominator,
                    0.0,
                )

            error = self.loss_(self.y_l_[val_idx], pred_fold)
            errors.append(error)
        return np.mean(errors)

    def _optimize_hyperparameters(self):

        bandwidth_candidates = (
            np.asarray(self.bandwidth_list)
            if self.bandwidth_list is not None
            else np.linspace(
                0.001,
                10.0,
                self.max_iter,
            )
        )

        method = self.opt_method.lower()

        if (
            method == "grad"
            and not self.kernel_.requires_grad
        ):
            method = "grid"

        optimizer = self.optimizer.lower()

        params = dict(self.optimizer_params or {})

        if method == "grad":

            if not OptimizerFactory.supports(
                optimizer,
                category="gradient",
            ):
                raise ValueError(
                    f"Optimizer '{optimizer}' "
                    f"does not support gradient optimization. "
                    f"Available: "
                    f"{OptimizerFactory.available_by_category('gradient')}"
                )

            params.update({
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
            })

        elif method == "grid":

            if not OptimizerFactory.supports(
                optimizer,
                category="search",
            ):
                raise ValueError(
                    f"Optimizer '{optimizer}' "
                    f"does not support search optimization. "
                    f"Available: "
                    f"{OptimizerFactory.available_by_category('search')}"
                )

            params.update({
                "param_grid": {
                    "bandwidth": bandwidth_candidates,
                },
                "random_state": self.random_state,
            })

        else:
            raise ValueError(
                f"Unknown optimization method: {method}"
            )

        self.optimizer_ = OptimizerFactory.create(
            optimizer,
            **params,
        )

        if method == "grad":
            result = self.optimizer_(
                self.kappa_cross_validation_error,
                init_param=np.array([1.0]),
            )
        else:
            result = self.optimizer_(
                self.kappa_cross_validation_error,
            )

        self.bandwidth_ = float(
            np.atleast_1d(result["x"])[0]
        )

        self.optimization_outputs_ = {
            "method": method,
            "optimizer": optimizer,
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
        split_ratio=0.5,
        overlap=0.0,
        as_predictions=False,
    ):
        ctx = resolve_training_context(
            X,
            y,
            X_l=X_l,
            y_l=y_l,
            as_predictions=as_predictions,
            split_ratio=split_ratio,
            overlap=overlap,
            random_state=self.random_state
        )
        self.X_k_ = ctx.X_k
        self.y_k_ = ctx.y_k
        self.X_l_ = ctx.X_l
        self.y_l_ = ctx.y_l
        self.as_predictions_ = ctx.as_predictions

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
        
        self.X_l_norm_, self.Y_l_norm_ = self._space_normalize(
            self.X_l_,
            prediction_space,
        )

        self._resolve_components()

        self.distance_matrix_ = self.distance_.matrix(
            self.Y_l_norm_,
            self.Y_l_norm_
        )

        self.cv_folds_ = list(self.cv_.split(self.X_l_norm_, self.Y_l_norm_))

        self._optimize_hyperparameters()

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.as_predictions_:
            prediction_space = X
        else:
            prediction_space = self._load_predictions(X)
        
        X_norm, Y_norm = self._space_normalize(
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

        preds = np.empty(K.shape[0], dtype=float)

        for i in range(K.shape[0]):
            w = K[i]
            
            if np.sum(w) <= 0:
                preds[i] = np.mean(self.y_l_)
            else:
                preds[i] = self.aggregator_.aggregate(
                    values=self.y_l_,
                    weights=w
                )

        return preds

