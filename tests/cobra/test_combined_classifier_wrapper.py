import numpy as np

from kfc_procedure.core.combiner.classification.combined_classifier import CobraClassifierCombiner


def test_cobra_wrapper_delegates_fit_predict_and_proba(monkeypatch):
    calls = {}

    class FakeCombinedClassifier:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, X, y, as_predictions=False):
            calls["fit_as_predictions"] = as_predictions
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    import kfc_procedure.core.combiner.classification.combined_classifier as module
    monkeypatch.setattr(module, "CombinedClassifier", FakeCombinedClassifier)

    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1])
    model = CobraClassifierCombiner(alpha=0.5).fit(X, y)

    assert calls["init"] == {"alpha": 0.5}
    assert calls["fit_as_predictions"] is True
    assert model.predict(X).shape == (2,)
    assert model.predict_proba(X).shape == (2, 2)
