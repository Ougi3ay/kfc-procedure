import numpy as np
import pytest

from kfc_procedure.cobra.core.kernels import KernelFactory
from kfc_procedure.cobra.core.kernels import (
    ReverseCoshKernel,
    ExponentialKernel,
    RadialKernel,
    CauchyKernel,
    EpanechnikovKernel,
    BiweightKernel,
    TriweightKernel,
    TriangularKernel,
    NaiveKernel,
    COBRAKernel,
)


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def D():
    return np.array([
        [0.0, 0.5, 1.0],
        [0.5, 0.0, 1.5],
        [1.0, 1.5, 0.0],
    ])


@pytest.fixture
def D_pos():
    return np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])


# =========================================================
# Factory tests
# =========================================================
def test_kernel_factory_create():
    k = KernelFactory.create("radial")
    assert isinstance(k, RadialKernel)


def test_kernel_factory_aliases():
    k1 = KernelFactory.create("gaussian")
    k2 = KernelFactory.create("rbf")

    assert type(k1) == type(k2)


def test_kernel_factory_available():
    names = KernelFactory.available()

    assert "radial" in names
    assert "cauchy" in names
    assert "cobra" in names


def test_factory_case_insensitive():
    k1 = KernelFactory.create("ExPoNeNtIaL")
    k2 = KernelFactory.create("exponential")

    assert type(k1) == type(k2)


# =========================================================
# Base kernel properties
# =========================================================
def test_kernel_params_update():
    k = ExponentialKernel(exponent=2.0)

    k.set_params(exponent=3.0)

    assert k.exponent == 3.0
    assert k.get_params()["exponent"] == 3.0


# =========================================================
# Reverse Cosh Kernel
# =========================================================
def test_reverse_cosh_kernel(D):
    k = ReverseCoshKernel(exponent=1.0)

    out = k(D)

    assert out.shape == D.shape
    assert np.all(out > 0)


# =========================================================
# Exponential Kernel
# =========================================================
def test_exponential_kernel(D):
    k = ExponentialKernel(exponent=1.0)

    out = k(D)

    assert out.shape == D.shape
    assert np.all(out <= 1.0)
    assert np.all(out >= 0.0)


# =========================================================
# Radial Kernel (RBF)
# =========================================================
def test_radial_kernel(D):
    k = RadialKernel()

    out = k(D)

    assert np.allclose(out, np.exp(-D))


# =========================================================
# Cauchy Kernel
# =========================================================
def test_cauchy_kernel(D_pos):
    k = CauchyKernel()

    out = k(D_pos)

    assert np.all(out > 0)
    assert np.all(out <= 1.0)


# =========================================================
# Compact kernels (support behavior)
# =========================================================
def test_epanechnikov_kernel():
    k = EpanechnikovKernel()

    D = np.array([[0.2, 1.2]])
    out = k(D)

    assert out[0, 0] > 0
    assert out[0, 1] == 0.0


def test_biweight_kernel():
    k = BiweightKernel()

    D = np.array([[0.5, 1.5]])
    out = k(D)

    assert out[0, 0] > 0
    assert out[0, 1] == 0.0


def test_triweight_kernel():
    k = TriweightKernel()

    D = np.array([[0.5, 1.5]])
    out = k(D)

    assert out[0, 0] > 0
    assert out[0, 1] == 0.0


def test_triangular_kernel():
    k = TriangularKernel()

    D = np.array([[0.3, 1.3]])
    out = k(D)

    assert out[0, 0] > 0
    assert out[0, 1] == 0.0


# =========================================================
# Naive kernel
# =========================================================
def test_naive_kernel(D):
    k = NaiveKernel()

    out = k(D)

    assert np.array_equal(out, D)


# =========================================================
# COBRA kernel
# =========================================================
def test_cobra_kernel():
    k = COBRAKernel(threshold=0.5)

    D = np.array([
        [0.2, 0.6],
        [0.4, 0.8],
    ])

    out = k(D)

    assert out[0, 0] == 1.0
    assert out[0, 1] == 0.0


# =========================================================
# Symmetry check (important for similarity kernels)
# =========================================================
def test_kernel_symmetry_property():
    k = RadialKernel()

    D = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    K = k(D)

    assert np.allclose(K, K.T)
