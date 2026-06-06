from __future__ import annotations

from typing import Any, List, Union

import numpy as np

from sklearn.base import (
    BaseEstimator as SkBaseEstimator,
    RegressorMixin,
)

from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
)

from kfc_procedure.cobra.core.adapters.base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)
from kfc_procedure.cobra.core.aggregators.base import (
    BaseAggregator,
    AggregatorFactory,
)
from kfc_procedure.cobra.core.distances.base import (
    BaseDistance,
    DistanceFactory,
)
from kfc_procedure.cobra.core.estimators.base import BaseEstimator
from kfc_procedure.cobra.core.kernels.base import (
    BaseKernel,
    KernelFactory,
)
from kfc_procedure.cobra.core.losses.base import (
    BaseLoss,
    LossFactory,
)
from kfc_procedure.cobra.core.optimizers.base import OptimizerFactory

from kfc_procedure.cobra.core.cv.base import (
    BaseCrossValidator,
    CVFactory
)
from kfc_procedure.cobra.utils.preprocessing import compute_normalization_constant, history_to_dataframe
from kfc_procedure.cobra.utils.resolve import (
    fit_estimators,
    predict_estimators,
    resolve_training_context
)


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
        learning_rate: float = 0.1,
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
        return predict_estimators(
            X=X,
            estimators=self.estimators_,
            n_jobs=self.n_jobs
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
                "one_parameter",
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
        bandwidth = float(np.atleast_1d(params)[0])
        
        self.adapter_.set_params(bandwidth=bandwidth)

        D = self.adapter_.transform(self.distance_matrix_)
        K = self.kernel_(D)

        errors = []

        for fold in self.cv_folds_:
            train_idx = fold.train_idx
            val_idx = fold.eval_idx

            K_val_train = K[np.ix_(val_idx, train_idx)]

            y_train = self.y_l_[train_idx]
            y_val = self.y_l_[val_idx]

            preds = self.aggregator_.aggregate_matrix(
                values=y_train,
                weights=K_val_train,
                fallback=0.0
            )

            error = self.loss_(y_val, preds)
            errors.append(error)
        return float(np.mean(errors))


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

        history_df = history_to_dataframe(
            result["history"],
            param_names=["bandwidth"],
        )

        self.optimization_outputs_ = {
            "method": method,
            "optimizer": optimizer,
            "bandwidth": self.bandwidth_,
            "score": result["score"],
            "history": history_df,
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
        self.global_mean_ = float(np.mean(self.y_l_))

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
        
        self.normalize_constant_ = (
            compute_normalization_constant(
                y=y,
                norm_constant=self.norm_constant,
                scale_factor=30.0,
                M=prediction_space.shape[1]
            )
        )
        
        self.Y_l_norm_ = (
            prediction_space
            * self.normalize_constant_
        )

        self._resolve_components()

        self.distance_matrix_ = self.distance_.matrix(
            self.Y_l_norm_,
            self.Y_l_norm_
        )

        self.cv_folds_ = list(self.cv_.split(self.X_l_, self.Y_l_norm_))

        self._optimize_hyperparameters()

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.as_predictions_:
            prediction_space = X
        else:
            prediction_space = self._load_predictions(X)
        
        Y_norm = (
            prediction_space
            * self.normalize_constant_
        )

        distance_matrix = self.distance_.matrix(
            Y_norm,
            self.Y_l_norm_,
        )

        self.adapter_.set_params(bandwidth=self.bandwidth_)

        D = self.adapter_.transform(distance_matrix)
        K = self.kernel_(D)

        preds = self.aggregator_.aggregate_matrix(
            values=self.y_l_,
            weights=K,
            fallback=self.global_mean_,
        )

        return preds

