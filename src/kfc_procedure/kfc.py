"""
KFCProcedure: Full three-stage ensemble learning pipeline.

This module implements the complete KFC (K-Step → F-Step → C-Step)
pipeline described in the KFC framework. The model combines:

    1. Divergence-aware clustering (K-step)
    2. Cluster-wise local learning (F-step)
    3. Ensemble aggregation (C-step)

The pipeline is designed to learn multiple representations of the
input space using Bregman divergences and fuse them into a final
predictive model.

Pipeline structure
------------------
The learning process is decomposed into three stages:

K-step (Clustering stage)
    - Applies multiple Bregman divergences
    - Produces divergence-specific cluster assignments

F-step (Local modeling stage)
    - Trains a local model per cluster per divergence
    - Learns specialized predictors on clustered data

C-step (Aggregation stage)
    - Combines divergence-level predictions
    - Produces final prediction via combiner strategy

Mathematical intuition
----------------------
Given data X and target y:

    1. Cluster space under divergences:
        C_d = KMeans_d(X)

    2. Local models per cluster:
        f_{d,k}(X_k) → y_k

    3. Aggregation:
        y = g( f_{d,1}(X), ..., f_{d,m}(X) )

This enables learning across multiple geometric assumptions induced
by different Bregman divergences.

Key advantages
--------------
- Multi-metric learning (multiple geometries)
- Local specialization per cluster
- Robust ensemble aggregation
- Compatible with sklearn API

Notes
-----
- Designed for both regression and classification tasks
- Uses train/test split internally to avoid overfitting in C-step
- Each stage is independently modular and replaceable
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from kfc_procedure.core.steps.cstep import CStep
from kfc_procedure.core.steps.fstep import FStep
from kfc_procedure.core.steps.kstep import KStep
from kfc_procedure.utils.logger import Logger


class KFCProcedure(BaseEstimator):
    """
    Full KFC pipeline estimator (sklearn-compatible).

    This estimator implements the complete KFCProcedure framework:

        K-step → F-step → C-step

    It performs divergence-aware clustering, cluster-wise local model
    fitting, and final ensemble aggregation.

    Parameters
    ----------
    divergences : list
        List of Bregman divergences used in K-step clustering.
        Can include strings or BaseBregmanDivergence instances.

    local_model : str or BaseLocalModel
        Base learner used in F-step for cluster-wise modeling.

    combiner : str or BaseCombiner
        Aggregation strategy used in C-step.

    divergences_params : dict, default=None
        Optional parameters for each divergence.

    local_model_params : dict, default=None
        Parameters for local models in F-step.

    combiner_params : dict, default=None
        Parameters for aggregation strategy in C-step.

    task : {"regression", "classification"}, default="regression"
        Learning task type.

    n_clusters : int, default=3
        Number of clusters per divergence.

    max_iter : int, default=300
        Maximum iterations for KMeans optimization.

    tol : float, default=1e-4
        Convergence tolerance for clustering.

    verbose : int, default=0
        Enable debugging output.
        Levels:
            0 - silent
            1 - basic info
            2 - detailed debug
            3 - trace (per-iteration / per-cluster)

    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    kstep_ : KStep
        Fitted divergence-aware clustering stage.

    fstep_ : FStep
        Fitted local model learning stage.

    cstep_ : CStep
        Fitted aggregation stage.

    Methods
    -------
    fit(X, y)
        Fit full KFC pipeline.

    predict(X)
        Predict using full pipeline.

    predict_proba(X)
        Predict probabilities (classification only).

    Notes
    -----
    The training procedure uses a train/test split internally:

        - K-step + F-step trained on training split
        - C-step trained on held-out predictions

    This reduces overfitting in the aggregation stage and improves
    generalization of the ensemble combiner.
    """

    def __init__(
        self,
        divergences,
        local_model,
        combiner,
        divergences_params: Optional[Dict] = None,
        local_model_params: Optional[Dict] = None,
        combiner_params: Optional[Dict] = None,
        task: str = "regression",
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self.divergences = divergences
        self.local_model = local_model
        self.combiner = combiner
        
        self.divergences_params = divergences_params
        self.local_model_params = local_model_params
        self.combiner_params = combiner_params
        
        self.task = task
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _make_logger(self) -> Logger:
        level = (
            self.verbose
            if isinstance(self.verbose, int)
            else int(bool(self.verbose))
        )
        
        return Logger(level)

    def _validate_task(self) -> None:
        """
        Validate estimator task.
        """
        if self.task not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                "task must be "
                "'regression' or 'classification'"
            )

    def fit(self, X: np.ndarray, y: np.ndarray):

        self._validate_task()

        X = np.asarray(X)
        y = np.asarray(y)

        self.logger_ = self._make_logger()

        self.logger_.info("KFC fit started")
        self.logger_.debug(f"X shape: {X.shape}, y shape: {y.shape}")

        X_k, X_l, y_k, y_l = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=self.random_state,
            stratify=y if self.task == "classification" else None
        )

        self.logger_.info("Train/test split completed")
        self.logger_.debug(f"X_k: {X_k.shape}, X_l: {X_l.shape}")

        self.kstep_ = KStep(
            divergences=self.divergences,
            divergences_params=self.divergences_params,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        self.logger_.info("Starting K-step clustering")

        self.kstep_.fit(X_k)

        self.logger_.debug(
            f"K-step done | divergences={len(self.divergences)} "
            f"| clusters={self.n_clusters}"
        )

        # training cluster assignments
        clusters_k = self.kstep_.clusters_

        # aggregation cluster assignments
        clusters_l = self.kstep_.predict(X_l)

        self.fstep_ = FStep(
            local_model=self.local_model,
            local_model_params=self.local_model_params,
            task=self.task,
            random_state=self.random_state
        )

        self.logger_.info("Starting F-step training")
        self.logger_.debug(f"Cluster keys: {list(clusters_k.keys())}")
        
        self.fstep_.fit(X_k, y_k, clusters_k)

        self.logger_.info("F-step completed")
        # M × K prediction matrix
        P_l = self.fstep_.predict(X_l, clusters_l)

        self.cstep_ = CStep(
            combiner=self.combiner,
            combiner_params=self.combiner_params,
            task=self.task,
            random_state=self.random_state
        )
        self.logger_.info("Starting C-step training")
        self.logger_.debug(f"P_l shape: {P_l.shape}")

        self.cstep_.fit(P_l, y_l)

        self.logger_.info("C-step completed")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(
            self,
            ["kstep_", "fstep_", "cstep_"]
        )

        X = np.asarray(X)

        self.logger_.info("Prediction started")
        # assign clusters
        clusters = self.kstep_.predict(X)

        self.logger_.debug("Cluster assignment done")
        # M × K prediction matrix
        P = self.fstep_.predict(X, clusters)
        self.logger_.debug(f"Prediction matrix: {P.shape}")
        
        # consensus aggregation
        out = self.cstep_.predict(P)

        self.logger_.info("Prediction finished")
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if self.task != "classification":
            raise AttributeError(
                "predict_proba only available for classification"
            )

        check_is_fitted(
            self,
            ["kstep_", "fstep_", "cstep_"]
        )

        X = np.asarray(X)

        clusters = self.kstep_.predict(X)

        P = self.fstep_.predict(X, clusters)

        return self.cstep_.predict_proba(P)

class KFCRegressor(KFCProcedure):
    """
    KFC estimator specialized for regression.

    Uses an explicit sklearn-compatible constructor.
    """

    def __init__(
        self,
        divergences,
        local_model,
        combiner,
        divergences_params: Optional[Dict] = None,
        local_model_params: Optional[Dict] = None,
        combiner_params: Optional[Dict] = None,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            divergences=divergences,
            local_model=local_model,
            combiner=combiner,
            divergences_params=divergences_params,
            local_model_params=local_model_params,
            combiner_params=combiner_params,
            task="regression",
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )


class KFCClassifier(KFCProcedure):
    """
    KFC estimator specialized for classification.

    Uses an explicit sklearn-compatible constructor.
    """

    def __init__(
        self,
        divergences,
        local_model,
        combiner,
        divergences_params: Optional[Dict] = None,
        local_model_params: Optional[Dict] = None,
        combiner_params: Optional[Dict] = None,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            divergences=divergences,
            local_model=local_model,
            combiner=combiner,
            divergences_params=divergences_params,
            local_model_params=local_model_params,
            combiner_params=combiner_params,
            task="classification",
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )