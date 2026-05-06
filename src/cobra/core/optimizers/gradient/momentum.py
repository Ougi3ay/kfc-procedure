
from __future__ import annotations
import numpy as np
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer


class MomentumGradientDescent(BaseGradientOptimizer):
    """Gradient descent with Nesterov momentum."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(self, objective, params):
        grad = self.gradient(objective, params)
        
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity - self.learning_rate * grad
        
        return params + self.velocity
    
    def __call__(self, objective, params):
        params = np.array(params, dtype=float)
        history = []
        
        for i in range(self.max_iter):
            grad = self.gradient(objective, params)
            params = self.step(objective, params)
            
            history.append(grad.copy())
            
            if np.linalg.norm(grad) < self.tol:
                break
        
        return params, history
