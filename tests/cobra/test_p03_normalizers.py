import numpy as np
import pytest

from kfc_procedure.cobra.core.normalizers import (
    StandardNormalizer,
    MinMaxNormalizer,
    NormalizerFactory,
)


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def sample_data():
    X = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ])
    return X


# =========================================================
# Standard Normalizer tests
# =========================================================
def test_standard_fit_transform(sample_data):
    norm = StandardNormalizer()

    Xn = norm.fit_transform(sample_data)

    # mean approx 0
    assert np.allclose(np.mean(Xn, axis=0), 0, atol=1e-7)

    # std approx 1
    assert np.allclose(np.std(Xn, axis=0), 1, atol=1e-7)


def test_standard_transform_consistency(sample_data):
    norm = StandardNormalizer()

    norm.fit(sample_data)
    X1 = norm.transform(sample_data)
    X2 = norm.transform(sample_data)

    assert np.allclose(X1, X2)


# =========================================================
# MinMax Normalizer tests
# =========================================================
def test_minmax_range(sample_data):
    norm = MinMaxNormalizer()

    Xn = norm.fit_transform(sample_data)

    assert np.all(Xn >= 0.0)
    assert np.all(Xn <= 1.0)


def test_minmax_fit_transform_equivalence(sample_data):
    norm = MinMaxNormalizer()

    X1 = norm.fit(sample_data).transform(sample_data)
    X2 = norm.fit_transform(sample_data)

    assert np.allclose(X1, X2)


# =========================================================
# Factory tests
# =========================================================
def test_factory_create_standard():
    norm = NormalizerFactory.create("standard")
    assert isinstance(norm, StandardNormalizer)


def test_factory_create_minmax():
    norm = NormalizerFactory.create("minmax")
    assert isinstance(norm, MinMaxNormalizer)


def test_factory_case_insensitive():
    n1 = NormalizerFactory.create("StAnDaRd")
    n2 = NormalizerFactory.create("standard")

    assert type(n1) == type(n2)


def test_factory_available():
    names = NormalizerFactory.available()

    assert "standard" in names
    assert "minmax" in names


# =========================================================
# Edge cases
# =========================================================
def test_constant_feature_minmax():
    X = np.array([
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
    ])

    norm = MinMaxNormalizer()
    Xn = norm.fit_transform(X)

    # should not crash (division stability)
    assert np.all(np.isfinite(Xn))


def test_constant_feature_standard():
    X = np.array([
        [3.0, 3.0],
        [3.0, 3.0],
        [3.0, 3.0],
    ])

    norm = StandardNormalizer()
    Xn = norm.fit_transform(X)

    assert np.all(np.isfinite(Xn))
