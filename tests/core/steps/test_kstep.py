import numpy as np
from sklearn.exceptions import NotFittedError
import pytest

from kfc_procedure.core.steps.kstep import KStep


def test_kstep_fit_and_predict_shapes():
    X = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
        [5.0, 5.0], [5.1, 5.0], [4.9, 5.1],
    ])
    step = KStep(divergences=["euclidean"], n_clusters=2, random_state=42)
    step.fit(X)

    assert set(step.clusters_) == {"euclidean"}
    assert step.clusters_["euclidean"].shape == (6,)
    pred = step.predict(X)
    assert pred["euclidean"].shape == (6,)
    assert set(np.unique(pred["euclidean"])).issubset({0, 1})


def test_kstep_predict_before_fit_raises():
    step = KStep(divergences=["euclidean"], n_clusters=2, random_state=42)
    with pytest.raises(NotFittedError):
        step.predict(np.zeros((2, 2)))
