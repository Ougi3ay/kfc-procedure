import numpy as np
import pytest

from kfc_procedure.cobra.core.adapters import KernelAdapterFactory
from kfc_procedure.cobra.core.adapters import (
    OneParameterKernelAdapter,
    TwoParameterKernelAdapter,
)


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def dist_matrix():
    return np.array([
        [0.0, 1.0],
        [1.0, 2.0],
    ])


@pytest.fixture
def dist_matrix_2():
    return np.array([
        [2.0, 3.0],
        [3.0, 4.0],
    ])


# =========================================================
# Factory tests
# =========================================================
def test_factory_create_one_param():
    adapter = KernelAdapterFactory.create("one_parameter", bandwidth=2.0)

    assert isinstance(adapter, OneParameterKernelAdapter)


def test_factory_create_two_param():
    adapter = KernelAdapterFactory.create("two_parameter")

    assert isinstance(adapter, TwoParameterKernelAdapter)


def test_factory_available():
    names = KernelAdapterFactory.available()

    assert "one_parameter" in names
    assert "two_parameter" in names


def test_factory_case_insensitive():
    a1 = KernelAdapterFactory.create("ONE_PARAMETER")
    a2 = KernelAdapterFactory.create("one_parameter")

    assert type(a1) == type(a2)


# =========================================================
# OneParameterKernelAdapter tests
# =========================================================
def test_one_parameter_scaling(dist_matrix):
    adapter = OneParameterKernelAdapter(bandwidth=3.0)

    out = adapter.transform(dist_matrix)

    assert np.allclose(out, dist_matrix * 3.0)


def test_one_parameter_shape_preserved(dist_matrix):
    adapter = OneParameterKernelAdapter(bandwidth=2.0)

    out = adapter.transform(dist_matrix)

    assert out.shape == dist_matrix.shape


def test_one_parameter_error():
    adapter = OneParameterKernelAdapter()

    with pytest.raises(ValueError):
        adapter.transform(np.array([[1.0]]), np.array([[2.0]]))


# =========================================================
# TwoParameterKernelAdapter tests
# =========================================================
def test_two_parameter_single_input(dist_matrix):
    adapter = TwoParameterKernelAdapter(alpha=2.0, beta=1.0)

    out = adapter.transform(dist_matrix)

    assert np.allclose(out, dist_matrix * 2.0)


def test_two_parameter_combination(dist_matrix, dist_matrix_2):
    adapter = TwoParameterKernelAdapter(alpha=1.0, beta=0.5)

    out = adapter.transform(dist_matrix, dist_matrix_2)

    expected = dist_matrix + 0.5 * dist_matrix_2

    assert np.allclose(out, expected)


def test_two_parameter_shape_mismatch():
    adapter = TwoParameterKernelAdapter()

    x = np.array([[1.0, 2.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        adapter.transform(x, y)


def test_two_parameter_too_many_inputs():
    adapter = TwoParameterKernelAdapter()

    with pytest.raises(ValueError):
        adapter.transform(
            np.array([[1.0]]),
            np.array([[2.0]]),
            np.array([[3.0]]),
        )


# =========================================================
# Parameter update tests
# =========================================================
def test_parameter_update():
    adapter = OneParameterKernelAdapter(bandwidth=1.0)

    adapter.set_params(bandwidth=5.0)

    assert adapter.bandwidth == 5.0
    assert adapter.get_params()["bandwidth"] == 5.0


def test_parameter_vector():
    adapter = TwoParameterKernelAdapter(alpha=2.0, beta=3.0)

    vec = adapter.parameter_vector()

    assert isinstance(vec, np.ndarray)
    assert len(vec) == 2
