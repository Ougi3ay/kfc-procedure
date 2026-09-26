import numpy as np
import pytest

from kfc_procedure.kfc import KFCProcedure


@pytest.mark.integration
def test_regression_pipeline_end_to_end(regression_data):
    X, y = regression_data
    model = KFCProcedure(
        divergences=["euclidean"],
        local_model="linear_regression",
        combiner="mean",
        task="regression",
        n_clusters=2,
        max_iter=50,
        random_state=42,
    )
    model.fit(X, y)
    pred = model.predict(X[:12])
    assert pred.shape == (12,)
    assert np.isfinite(pred).all()
