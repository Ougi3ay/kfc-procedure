from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import copy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.steps.cstep import BaseCStep, CStep
from kfc_procedure.core.steps.fstep import BaseFStep, FStep
from kfc_procedure.core.steps.kstep import BaseKStep, KStep


class BaseKFC(BaseEstimator, ABC):
    """
    Base class for KFC (K-means, Fit, Combine) meta-estimators.

    The KFC procedure is a modular ensemble learning pipeline composed of
    three sequential steps:

    1. **K-step (Clustering)**:
       Partition the feature space using one or multiple divergence-based
       clustering strategies.

    2. **F-step (Local Fitting)**:
       Train local predictive models within each cluster.

    3. **C-step (Combination)**:
       Aggregate predictions from all local models into a final output.

    This class supports flexible configuration via dictionaries or
    pre-instantiated step objects.

    Parameters
    ----------
    kstep : BaseKStep or dict
        Configuration or instance of the clustering step.
        Example:
        {
            "divergences": [
                {"name": "euclidean"},
                {"name": "mahalanobis", "params": {...}}
            ]
        }

    fstep : BaseFStep or dict
        Configuration or instance of the local model fitting step.

    cstep : BaseCStep or dict
        Configuration or instance of the aggregation step.

    n_clusters : int, default=8
        Default number of clusters used in K-step. Applied to all divergences
        unless explicitly overridden.

    max_iter : int, default=300
        Maximum number of iterations for clustering algorithms.

    tol : float, default=1e-4
        Convergence tolerance for clustering.

    verbose : bool, default=False
        If True, enables verbose logging.

    random_state : int or None, default=None
        Random seed used for reproducibility and internal data splitting.

    Notes
    -----
    - Top-level parameters (`n_clusters`, `max_iter`, etc.) are automatically
      propagated to each divergence configuration unless explicitly defined.
    - This design follows a "global defaults with local override" pattern.
    """

    def __init__(
        self,
        kstep: Union[BaseKStep, Dict],
        fstep: Union[BaseFStep, Dict],
        cstep: Union[BaseCStep, Dict],
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.kstep = kstep
        self.fstep = fstep
        self.cstep = cstep
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: np.ndarray | None = None,
    ) -> "BaseKFC":
        """
        Fit the KFC pipeline.

        The training process consists of:
        1. Splitting the dataset into two subsets:
           - Pre-fit set (for K-step and F-step)
           - Aggregation set (for C-step)
        2. Learning cluster structure on the pre-fit set.
        3. Training local models per cluster.
        4. Generating predictions on the held-out set.
        5. Training the aggregation model (C-step).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        y : np.ndarray of shape (n_samples,)
            Target values.

        stratify : np.ndarray, optional
            Stratification labels for the internal split (useful for classification).

        Returns
        -------
        self : BaseKFC
            Fitted estimator.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape={X.shape}.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples "
                f"(got X={X.shape[0]}, y={y.shape[0]})."
            )

        # Resolve pipeline components
        self._resolve_steps()

        # Split data
        X_pre, X_agg, y_pre, y_agg = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=self.random_state,
            stratify=stratify,
        )

        # K-step
        self.kstep_.fit(X_pre)
        clusters_pre = self.kstep_.predict(X_pre)

        # F-step
        self.fstep_.fit(X_pre, y_pre, clusters_pre)

        # Generate predictions for aggregation
        clusters_agg = self.kstep_.predict(X_agg)
        predictions = self.fstep_.predict(X_agg, clusters_agg)

        X_cstep = np.hstack(list(predictions.values()))

        # C-step
        self.cstep_.fit(X_cstep, y_agg)

        self.is_fitted_ = True
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted KFC model."""
        ...

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """
        Generate intermediate prediction matrix from F-step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_cstep : np.ndarray
            Matrix of stacked predictions used by C-step.
        """
        check_is_fitted(self, ["kstep_", "fstep_", "cstep_"])

        clusters = self.kstep_.predict(X)
        predictions = self.fstep_.predict(X, clusters)

        return np.hstack(list(predictions.values()))

    def _resolve_steps(self) -> None:
        """
        Resolve configuration dictionaries into concrete step instances.

        This method:
        - Instantiates K-step, F-step, and C-step if given as dictionaries
        - Injects global parameters into divergence configurations
        """
        # Kstep
        if isinstance(self.kstep, BaseKStep):
            self.kstep_ = self.kstep
        else:
            kstep_config = copy.deepcopy(self.kstep)
            self._inject_kstep_defaults(kstep_config)
            self.kstep_ = KStep(kstep_config)
        # Fstep
        if isinstance(self.fstep, BaseFStep):
            self.fstep_ = self.fstep
        else:
            self.fstep_ = FStep(self.fstep)

        # Cstep
        if isinstance(self.cstep, BaseCStep):
            self.cstep_ = self.cstep
        else:
            self.cstep_ = CStep(self.cstep)

    def _inject_kstep_defaults(self, config: Dict[str, Any]) -> None:
        """
        Inject top-level default parameters into each divergence.

        Parameters
        ----------
        config : dict
            K-step configuration dictionary (modified in-place).

        Notes
        -----
        This ensures that global parameters such as `n_clusters` are applied
        to all divergences unless explicitly overridden.
        """
        divergences = config.get("divergences", [])

        normalized_divs = []

        for div in divergences:
            # Allow shorthand: "euclidean"
            if isinstance(div, str):
                div = {"name": div}

            params = div.setdefault("params", {})

            # Inject defaults (do not override user-defined values)
            params.setdefault("n_clusters", self.n_clusters)
            params.setdefault("max_iter", self.max_iter)
            params.setdefault("tol", self.tol)
            params.setdefault("random_state", self.random_state)

            normalized_divs.append(div)

        config["divergences"] = normalized_divs
