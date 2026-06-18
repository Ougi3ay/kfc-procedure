import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from kfc_procedure import KFCClassifier, KFCRegressor


def test_kfc_regressor_smoke_runs_end_to_end():
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    model = KFCRegressor(
        divergences=["euclidean"],
        local_model="linear_regression",
        combiner="mean",
        n_clusters=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(mean_squared_error(y_test, pred))


def test_kfc_classifier_smoke_runs_end_to_end():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    model = KFCClassifier(
        divergences=["euclidean"],
        local_model="logistic_regression",
        combiner="majority_vote",
        n_clusters=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert pred.shape == y_test.shape
    assert np.issubdtype(pred.dtype, np.number)
    assert accuracy_score(y_test, pred) >= 0.0
