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
from cobra.utils.resolve import fit_estimators, predict_estimators, resolve_training_context


class CombineClassifier(ABC, SkBaseEstimator):

    def __init__(
        self,
        estimators: List[Union[str, BaseEstimator]] | None = None,
        estimators_params: Dict[str, Any] | None = None,
        distance: str = "hamming",
        distance_params: Dict[str, Any] | None = None,
        kernel: str = "rbf",
        kernel_params: Dict[str, Any] | None = None,
        aggregator: str = "weighted_vote",
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

    # -------------------------
    # BASE MODELS
    # -------------------------
    def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):

        default_estimators = [
            "logistic_regression",
            "decision_tree_classifier",
            "svc",
            "k_neighbors_classifier",
        ]

        estimators = self.estimators or default_estimators

        return fit_estimators(
            X=X_k,
            y=y_k,
            estimators_params=self.estimators_params,
            estimators=estimators,
            n_jobs=self.n_jobs,
        )

    def _load_predictions(self, X: np.ndarray) -> np.ndarray:
        return predict_estimators(
            X=X,
            estimators=self.estimators_,
            n_jobs=self.n_jobs,
        )

    # -------------------------
    # COMPONENTS
    # -------------------------
    def _resolve_components(self):

        self.distance_: BaseDistance = DistanceFactory.create(
            self.distance,
            **(self.distance_params or {}),
        )

        self.kernel_: BaseKernel = KernelFactory.create(
            self.kernel,
            **(self.kernel_params or {}),
        )

        self.loss_: BaseLoss = LossFactory.create(
            self.loss,
            **(self.loss_params or {}),
        )

        self.cv_: BaseCrossValidator = CVFactory.create(
            "kfold",
            n_splits=self.n_cv,
            shuffle=True,
            random_state=self.random_state,
        )

        self.adapter_: BaseKernelAdapter = KernelAdapterFactory.create(
            "one_parameter",
            bandwidth=1.0,
        )

        self.aggregator_: BaseAggregator = AggregatorFactory.create(
            self.aggregator,
            **(self.aggregator_params or {}),
        )

    # -------------------------
    # OPTIMIZER (RESTORED)
    # -------------------------
    def _optimize_hyperparameters(self):

        bandwidth_candidates = (
            np.asarray(self.bandwidth_list)
            if self.bandwidth_list is not None
            else np.linspace(0.001, 10.0, self.max_iter)
        )

        params = dict(self.optimizer_params or {})
        params.update({
            "param_grid": {"bandwidth": bandwidth_candidates},
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        })

        self.optimizer_ = OptimizerFactory.create(
            self.optimizer,
            **params,
        )

        result = self.optimizer_(self.kappa_cross_validation_error)

        self.bandwidth_ = float(np.atleast_1d(result["x"])[0])

        self.optimization_outputs_ = {
            "method": "grid",
            "optimizer": self.optimizer,
            "bandwidth": self.bandwidth_,
            "score": result["score"],
            "history": result["history"],
        }

    def kappa_cross_validation_error(self, params):

        bandwidth = float(np.atleast_1d(params)[0])

        self.adapter_.set_params(bandwidth=bandwidth)

        D = self.adapter_.transform(self.distance_matrix_)
        K = self.kernel_(D)

        errors = []

        for fold in self.cv_folds_:

            train_idx = fold.train_idx
            val_idx = fold.eval_idx

            K_vt = K[np.ix_(val_idx, train_idx)]
            y_train = self.y_l_[train_idx]

            preds = []

            for i in range(len(val_idx)):
                w = K_vt[i]

                if np.sum(w) <= 0:
                    pred = self.global_majority_class_
                else:
                    pred = self.aggregator_.aggregate(y_train, w)

                preds.append(pred)

            preds = np.array(preds)
            y_true = self.y_l_[val_idx]
            error = self.loss_(y_true, preds)
            errors.append(error)

        return np.mean(errors)

    # -------------------------
    # FIT
    # -------------------------
    def fit(self, X, y, X_l=None, y_l=None, split_ratio=0.5, overlap=False, as_predictions=False):

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

        self.X_k_, self.y_k_ = ctx.X_k, ctx.y_k
        self.X_l_, self.y_l_ = ctx.X_l, ctx.y_l
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

    # -------------------------
    # PREDICT (FIXED)
    # -------------------------
    def predict(self, X):

        X = check_array(X)

        preds_space = X if self.as_predictions_ else self._load_predictions(X)

        D = self.distance_.matrix(preds_space, self.pred_l)

        self.adapter_.set_params(bandwidth=self.bandwidth_)
        D = self.adapter_.transform(D)

        K = self.kernel_(D)

        outputs = []

        for i in range(K.shape[0]):
            w = K[i]

            if np.sum(w) <= 0:
                outputs.append(self.global_majority_class_)
            else:
                outputs.append(
                    self.aggregator_.aggregate(self.y_l_, w)
                )

        return np.array(outputs)

    # -------------------------
    # PROBA
    # -------------------------
    def predict_proba(self, X):

        X = check_array(X)

        preds_space = X if self.as_predictions_ else self._load_predictions(X)

        D = self.distance_.matrix(preds_space, self.pred_l)

        self.adapter_.set_params(bandwidth=self.bandwidth_)
        D = self.adapter_.transform(D)

        K = self.kernel_(D)

        classes = self.classes_
        proba = np.zeros((len(K), len(classes)))

        for i in range(len(K)):
            w = K[i]

            if np.sum(w) <= 0:
                proba[i, np.where(classes == self.global_majority_class_)[0][0]] = 1.0
                continue

            proba[i] = self.aggregator_.aggregate_proba(
                values=self.y_l_,
                weights=w,
                classes=classes
            )

        return proba
