
from __future__ import annotations
from abc import ABC
from typing import Callable, Dict, Any
import numpy as np

from cobra.core.optimizers.base import BaseOptimizer


class BaseSearchOptimizer(BaseOptimizer, ABC):
    """
    Base class for derivative-free optimization.
    """

    def __init__(
        self,
        bounds: Dict[str, tuple] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.bounds = bounds
