import numpy as np
import pytest

from kfc_procedure.kfc import KFCProcedure


@pytest.mark.integration
def test_classification_pipeline_end_to_end(classification_data):
    X, y = classification_data
    model = KFCProcedure(
        divergences=["euclidean"],
        local_model="logistic_regression",
        combiner="stacking_classifier",
        task="classification",
        n_clusters=2,
        max_iter=50,
        random_state=42,
    )
    model.fit(X, y)
    pred = model.predict(X[:12])
    assert pred.shape == (12,)


@pytest.mark.integration
def test_classification_predict_proba_end_to_end(classification_data):
    X, y = classification_data
    model = KFCProcedure(
        divergences=["euclidean"],
        local_model="logistic_regression",
        combiner="stacking_classifier",
        task="classification",
        n_clusters=2,
        max_iter=50,
        random_state=42,
    )
    model.fit(X, y)
    proba = model.predict_proba(X[:12])
    assert proba.shape == (12, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
