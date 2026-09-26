import numpy as np
import pytest
from sklearn.base import clone

from kfc_procedure.core.combiner.classification.majority_vote import MajorityVoteCombiner
from kfc_procedure.core.combiner.classification.stacking import StackingClassifierCombiner


def test_majority_vote():
    X = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 2]])
    model = MajorityVoteCombiner().fit(X)
    np.testing.assert_array_equal(model.predict(X), np.array([0, 1, 2], dtype=object))


def test_majority_vote_accepts_random_state_contract():
    model = MajorityVoteCombiner(random_state=7)
    assert model.random_state == 7
    assert clone(model).random_state == 7


def test_stacking_classifier_predict_shape():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1])
    model = StackingClassifierCombiner().fit(X, y)
    assert model.predict(X).shape == (6,)


def test_stacking_classifier_predict_proba_if_supported():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1])
    model = StackingClassifierCombiner().fit(X, y)
    if not hasattr(model, "predict_proba"):
        pytest.fail("StackingClassifierCombiner should expose predict_proba()")
    proba = model.predict_proba(X)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)
