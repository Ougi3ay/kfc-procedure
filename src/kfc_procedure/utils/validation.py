
from __future__ import annotations
from typing import Literal
import numpy as np


def apply_numerical_policy(
    X: np.ndarray,
    *,
    warn: bool = False,
    policy: Literal["auto", "always", "never"]
) -> np.ndarray:
    eps   = 1e-12

    if policy == "never":
        return X
    
    neg = X < 0
    if not np.any(neg):
        return X
    
    if policy == "always":
        if warn:
            print(f"[numerics] clamped {np.sum(neg)} negative values")
        return np.maximum(X, 0.0)
    
    if policy == "auto":
        small = neg & (X >= -eps)
        large = neg & (X < -eps)

        if np.any(small):
            if policy.warn:
                print(f"[numerics] clamped {np.sum(small)} small negatives to 0")
            X[small] = 0.0

        if np.any(large):
            raise ValueError(
                f"Numerical instability detected in divergence: "
                f"{np.min(X):.3e} < -eps ({-eps:.3e})."
            )
        
        return X

    raise ValueError(f"Unknown clamp mode: {policy!r}. Must be 'auto', 'always', or 'never'.")

