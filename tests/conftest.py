import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def prediction_matrix():
    return np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])


@pytest.fixture
def regression_targets():
    return np.array([1.5, 3.0, 4.5])


@pytest.fixture
def regression_data():
    return make_regression(
        n_samples=120,
        n_features=5,
        noise=0.1,
        random_state=42,
    )


@pytest.fixture
def classification_data():
    return make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
