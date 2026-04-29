"""
Base classes for the KFC procedure.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.steps.cstep import BaseCStep, CStep
from kfc_procedure.core.steps.fstep import BaseFStep, FStep
from kfc_procedure.core.steps.kstep import BaseKStep, KStep

class BaseKFC(BaseEstimator, ABC):
    """
    KFC (K-means, Fitting, Combining) meta-estimator.

    The KFC pipeline is a modular ensemble strategy composed of:

    - K-step: clustering multiple divergences/configurations
    - F-step: fitting local models on cluster partitions
    - C-step: aggregating held-out predictions

    Parameters
    ----------
    kstep : BaseKStep | list[dict]
        K-step configuration or pre-instantiated clustering component.

    fstep : BaseFStep | dict
        F-step configuration or pre-instantiated local fitting component.

    cstep : BaseCStep | dict
        C-step aggregation configuration or pre-instantiated aggregator.

    random_state : int | None, default=None
        Random seed used for the internal train/test split.
    """

    def __init__(
        self,
        kstep: Union[BaseKStep, List[Dict]],
        fstep: Union[BaseFStep, Dict],
        cstep: Union[BaseCStep, Dict],
        random_state: int | None = None
    ):
        self.kstep = kstep
        self.fstep = fstep
        self.cstep = cstep
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray, stratify: np.ndarray | None = None, *args, **kwargs) -> "BaseKFC":
        """
        Fit the KFC pipeline using the provided training data.

        This method splits the data into pre-fit and aggregation partitions,
        trains the K-step and F-step components on the first partition, then
        fits the C-step using held-out predictions from the second partition.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        y : np.ndarray
            Target values or labels of shape (n_samples,).

        stratify : np.ndarray | None, default=None
            Stratification labels used for the internal split.

        *args
            Additional positional arguments forwarded to subclasses.

        **kwargs
            Additional keyword arguments forwarded to subclasses.

        Returns
        -------
        BaseKFC
            Fitted KFC estimator.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples "
                f"(got X={X.shape[0]}, y={y.shape[0]})."
            )
        
        # resolve
        self.resolve()

        # split
        X_pre, X_agg, y_pre, y_agg = train_test_split(
            X, y, test_size=0.5, random_state=self.random_state, stratify=stratify
        )
        self.kstep_.fit(X_pre)
        clusters = self.kstep_.predict(X_pre)
        
        self.fstep_.fit(X_pre, y_pre, clusters)
        predictions = self.fstep_.predict(X_agg, clusters)
        
        X_cstep = np.hstack(list(predictions.values()))

        self.cstep_.fit(X_cstep, y_agg)

        self.is_fitted_ = True
        return self
    
    @abstractmethod
    def predict(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict using the fitted KFC model."""
        ...
    
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal method to get the prediction matrix from F-step."""
        check_is_fitted(self, ["kstep_", "fstep_", "cstep_"])
        clusters    = self.kstep_.predict(X)
        predictions = self.fstep_.predict(X, clusters)
        X_cstep = np.hstack(list(predictions.values()))
        return X_cstep

    
    def resolve(self):
        """
        Resolve configuration dictionaries into concrete pipeline components.

        If a component is already instantiated, it is kept as-is. Otherwise,
        the corresponding factory wrapper is used to build the component from
        its configuration dictionary.
        """
        if isinstance(self.kstep, BaseKStep):
            self.kstep_ = self.kstep
        else:
            self.kstep_ = KStep(self.kstep)
        
        if isinstance(self.fstep, BaseFStep):
            self.fstep_ = self.fstep
        else:
            self.fstep_ = FStep(self.fstep)
        
        if isinstance(self.cstep, BaseCStep):
            self.cstep_ = self.cstep
        else:
            self.cstep_ = CStep(self.cstep)
