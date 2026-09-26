from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory


@CombinerFactory.register(
    "stacking_classifier",
    categories={"classification"},
)
class StackingClassifierCombiner(BaseCombiner):
    """
    Supervised stacking combiner for classification.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        super().__init__(
            random_state=random_state,
        )

        self.C = C
        self.max_iter = max_iter

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        X = np.asarray(X)
        y = np.asarray(y)

        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_

        return self

    def combine(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        check_is_fitted(
            self,
            "model_",
        )

        X = np.asarray(X)

        return self.model_.predict(X)

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        check_is_fitted(
            self,
            "model_",
        )

        X = np.asarray(X)

        return self.model_.predict_proba(X)
