"""
Distance utils
"""
import numba as nb
import numpy as np

@nb.jit(nopython=True, parallel=True, fastmath=True)
def hamming_matrix_numba(x: np.ndarray, y: np.ndarray):
    n_x, n_y = x.shape[0], y.shape[0]
    n_features = x.shape[1]
    distances = np.empty((n_x, n_y), dtype=np.float64)

    for i in nb.prange(n_x):
        for j in range(n_y):
            diff_count = 0.0
            for k in range(n_features):
                if x[i, k] != y[j, k]:
                    diff_count += 1.0
            distances[i, j] = diff_count / n_features  
    return distances