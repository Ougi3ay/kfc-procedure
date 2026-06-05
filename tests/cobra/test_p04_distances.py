import numpy as np
import pytest

from cobra.core.distances import DistanceFactory


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def sample_data():
    x = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    y = np.array([
        [1.0, 2.0],
        [4.0, 8.0],
    ])

    return x, y


# =========================================================
# Factory tests
# =========================================================
def test_factory_create_all():
    names = DistanceFactory.available()

    assert "euclidean" in names
    assert "l1" in names or "manhattan" in names
    assert "cosine" in names
    assert "hamming" in names


def test_factory_case_insensitive():
    d1 = DistanceFactory.create("EuClIdEaN")
    d2 = DistanceFactory.create("euclidean")

    assert type(d1) == type(d2)


# =========================================================
# Shape correctness
# =========================================================
def test_distance_matrix_shape(sample_data):
    x, y = sample_data

    dist = DistanceFactory.create("euclidean")
    D = dist.matrix(x, y)

    assert D.shape == (len(x), len(y))


# =========================================================
# Euclidean tests
# =========================================================
def test_euclidean_self_zero():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    dist = DistanceFactory.create("euclidean")
    D = dist.matrix(x, x)

    assert np.allclose(np.diag(D), 0.0)


# =========================================================
# Manhattan tests
# =========================================================
def test_manhattan_non_negative(sample_data):
    x, y = sample_data

    dist = DistanceFactory.create("manhattan")
    D = dist.matrix(x, y)

    assert np.all(D >= 0)


# =========================================================
# Minkowski tests
# =========================================================
def test_minkowski_consistency():
    x = np.random.rand(10, 3)
    y = np.random.rand(8, 3)

    dist = DistanceFactory.create("minkowski", p=2)
    D = dist.matrix(x, y)

    assert D.shape == (10, 8)
    assert np.all(np.isfinite(D))


# =========================================================
# Cosine tests
# =========================================================
def test_cosine_range():
    x = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    y = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    dist = DistanceFactory.create("cosine")
    D = dist.matrix(x, y)

    assert np.all(D >= 0)
    assert np.all(np.isfinite(D))


# =========================================================
# Hamming tests
# =========================================================
def test_hamming_binary():
    x = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ])

    y = np.array([
        [1, 0, 0],
        [0, 1, 1],
    ])

    dist = DistanceFactory.create("hamming")
    D = dist.matrix(x, y)

    assert D.shape == (2, 2)
    assert np.all(D >= 0)
    assert np.all(D <= 1)


# =========================================================
# Stability tests
# =========================================================
def test_distance_no_nan():
    x = np.zeros((5, 3))
    y = np.zeros((5, 3))

    for name in DistanceFactory.available():
        dist = DistanceFactory.create(name)
        D = dist.matrix(x, y)

        assert np.all(np.isfinite(D))

def test_euclidean_exact_value():
    x = np.array([[0.0, 0.0]])
    y = np.array([[3.0, 4.0]])
    dist = DistanceFactory.create("euclidean")
    D = dist.matrix(x, y)
    assert np.isclose(D[0, 0], 5.0)

def test_manhattan_exact():
    x = np.array([[1.0, 2.0]])
    y = np.array([[4.0, 6.0]])
    dist = DistanceFactory.create("manhattan")
    D = dist.matrix(x, y)
    assert D[0, 0] == 7.0

def test_minkowski_p2_equals_euclidean():
    x = np.random.rand(5, 3)
    y = np.random.rand(4, 3)
    d1 = DistanceFactory.create("minkowski", p=2).matrix(x, y)
    d2 = DistanceFactory.create("euclidean").matrix(x, y)
    assert np.allclose(d1, d2)

def test_hamming_binary():
    x = np.array([[1, 0, 1]])
    y = np.array([[1, 1, 0]])
    dist = DistanceFactory.create("hamming")
    D = dist.matrix(x, y)
    assert 0.0 <= D[0, 0] <= 1.0

def test_hamming_self_zero():
    x = np.array([[1, 0, 1], [0, 1, 0]])
    dist = DistanceFactory.create("hamming")
    D = dist.matrix(x, x)
    assert np.allclose(np.diag(D), 0.0)

def test_symmetry():
    x = np.random.rand(5, 3)
    y = np.random.rand(6, 3)
    dist = DistanceFactory.create("euclidean")
    D1 = dist.matrix(x, y)
    D2 = dist.matrix(y, x)
    assert np.allclose(D1, D2.T)