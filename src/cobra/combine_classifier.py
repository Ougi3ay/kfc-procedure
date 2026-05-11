
from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Union
import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.utils import check_array

from cobra.core.adapters.base import BaseKernelAdapter, KernelAdapterFactory
from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator
from cobra.core.kernels.base import BaseKernel, KernelFactory
from cobra.core.losses.base import BaseLoss, LossFactory
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.validators.base import BaseCrossValidator, CVFactory
from cobra.utils.resolve import predict_estimators, fit_estimators, resolve_training_context

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class CombineClassifier(ABC, SkBaseEstimator):
    def __init__(
        self,
        estimators: List[Union[str, BaseEstimator]] | None = None,
        estimators_params: Dict[str, Any] | None = None,
        distance: str = "hamming",
        distance_params: Dict[str, Any] | None = None,
        kernel: str = "indicator",
        kernel_params: Dict[str, Any] | None = None,
        aggregator: str = "majority_vote",
        aggregator_params: Dict[str, Any] | None = None,
        loss: str = "mse",
        loss_params: dict[str, Any] | None = None,
        optimizer: str = "grid",
        optimizer_params: dict[str, Any] | None = None,
        n_jobs: int = 1,
        bandwidth_list: np.ndarray | None = None,
        max_iter: int = 300,
        n_cv: int = 5,
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
        self.n_jobs = n_jobs
        self.bandwidth_list = bandwidth_list
        self.max_iter = max_iter
        self.n_cv = n_cv
        self.random_state = random_state

    def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):
        default_estimators = [
            "logistic_regression",
            "decision_tree_classifier",
            "svc",
            "k_neighbors_classifier",
        ]
        estimators = self.estimators or default_estimators
        machines = fit_estimators(
            X=X_k,
            y=y_k,
            estimators_params=self.estimators_params,
            estimators=estimators,
            n_jobs=self.n_jobs
        )
        return machines

    def _load_predictions(self, X: np.ndarray) -> np.ndarray:
        return predict_estimators(
            X=X,
            estimators=self.estimators_,
            n_jobs=self.n_jobs
        )
    
    def _resolve_components(self):
        self.distance_: BaseDistance = DistanceFactory.create(
            self.distance,
            **(self.distance_params or {}),
        )

        self.kernel_: BaseKernel = KernelFactory.create(
            self.kernel,
            **(self.kernel_params or {}),
        )

        self.loss_ : BaseLoss = LossFactory.create(
            self.loss,
            **(self.loss_params or {})
        )

        self.cv_ : BaseCrossValidator = (
            CVFactory.create(
                "kfold",
                n_splits=self.n_cv,
                shuffle=True,
                random_state=self.random_state,
            )
        )

        self.adapter_: BaseKernelAdapter = KernelAdapterFactory.create(
            "one_parameter",
            bandwidth=1.0,
        )

        agg_params = self.aggregator_params or {}
        agg_params["classes"] = self.classes_
        self.aggregator_: BaseAggregator = AggregatorFactory.create(
            self.aggregator,
            **agg_params,
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

        if not OptimizerFactory.supports(
            self.optimizer,
            category="search"
        ):
            raise ValueError(
                f"Optimizer '{self.optimizer}' "
                f"does not support search optimization. "
                f"Available: "
                f"{OptimizerFactory.available_by_category('search')}"
            )
        params = dict(self.optimizer_params or {})
        params.update({
            "param_grid": {
                "bandwidth": bandwidth_candidates,
            },
            "random_state": self.random_state,
        })

        self.optimizer_ = OptimizerFactory.create(
            self.optimizer,
            **params,
        )
        result = self.optimizer_(self.kappa_cross_validation_error)

        self.bandwidth_ = float(
            np.atleast_1d(result["x"])[0]
        )

        self.optimization_outputs_ = {
            "method": "grid",
            "optimizer": self.optimizer,
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
        overlap=False,
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
            self.classes_ = np.unique(self.y_k_)
            self.estimators_ = self._fit_estimators(self.X_k_, self.y_k_)
            self.pred_l = self._load_predictions(self.X_l_)
        else:
            self.classes_ = np.unique(self.y_l_)
            self.pred_l = self.X_l_

        classes, counts = np.unique(self.y_l_, return_counts=True)
        self.global_majority_class_ = classes[np.argmax(counts)]

        self._resolve_components()

        self.distance_matrix_ = self.distance_.matrix(self.pred_l, self.pred_l)

        self.cv_folds_ = list(self.cv_.split(self.X_l_, self.y_l_))

        self._optimize_hyperparameters()
        return self
    
    def predict(self, X: np.ndarray):
        X = check_array(X)

        predictions = (
            X if self.as_predictions_
            else self._load_predictions(X)
        )

        distance_matrix = self.distance_.matrix(predictions, self.pred_l)

        self.adapter_.set_params(bandwidth=self.bandwidth_)
        D = self.adapter_.transform(distance_matrix)
        K = self.kernel_(D)

        preds = np.empty(K.shape[0], dtype=float)

        for i in range(K.shape[0]):
            w = K[i]

            if np.sum(w) <= 0:
                preds[i] = self.global_majority_class_
            else:
                preds[i] = self.aggregator_.aggregate(
                    values=self.y_l_,
                    weights=w
                )

        return preds
    
    def predict_proba(self, X: np.ndarray):
        X = check_array(X)

        predictions = (
            X if self.as_predictions_
            else self._load_predictions(X)
        )

        distance_matrix = self.distance_.matrix(predictions, self.pred_l)

        self.adapter_.set_params(bandwidth=self.bandwidth_)
        D = self.adapter_.transform(distance_matrix)
        K = self.kernel_(D)

        n_samples = K.shape[0]
        classes = self.classes_

        proba = np.zeros((n_samples, len(classes)), dtype=float)

        class_masks = {
            c: (self.y_l_ == c)
            for c in classes
        }

        for i in range(n_samples):
            w = K[i]
            denom = np.sum(w)

            if denom <= 0:
                proba[i, np.argmax(classes == self.global_majority_class_)] = 1.0
                continue

            for j, c in enumerate(classes):
                proba[i, j] = np.sum(w[class_masks[c]]) / denom

        return proba
        