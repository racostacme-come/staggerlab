import numpy as np
import pytest
from numpy.testing import assert_allclose

from staggerlab import Grid, _core, project


@pytest.mark.parametrize("shape", [(7, 11), (16, 12), (3, 3)])
def test_adjoint_composition_and_conservation(shape):
    g = Grid(*shape, lx=2.3, ly=4.1)
    p, u, v = np.random.default_rng(76).normal(size=(3, *g.shape))
    gx, gy = g.gradient(p)
    assert_allclose(g.divergence(gx, gy), g.laplacian(p), atol=2e-12)
    assert abs(g.inner(p, g.divergence(u, v)) + g.inner(gx, u) + g.inner(gy, v)) < 2e-12
    assert abs(np.sum(g.divergence(u, v))) < 2e-12
    assert_allclose(g.laplacian(np.ones(g.shape)), 0, atol=1e-14)


@pytest.mark.parametrize("shape", [(9, 7), (16, 12)])
def test_projection_idempotent_orthogonal_preserves_means(shape):
    g = Grid(*shape, lx=3.7, ly=2.2)
    u, v = np.random.default_rng(14).normal(size=(2, *g.shape))
    before = u.copy(), v.copy()
    p = project(g, u, v)
    p2 = project(g, p.u, p.v)
    assert_allclose(g.divergence(p.u, p.v), 0, atol=5e-13)
    assert_allclose(p2.u, p.u, atol=3e-14)
    assert_allclose(p2.v, p.v, atol=3e-14)
    assert_allclose([p.u.mean(), p.v.mean()], [u.mean(), v.mean()], atol=2e-16)
    assert abs(p.potential.mean()) < 1e-16
    assert abs(p.orthogonality_defect) < 2e-13
    gx, gy = g.gradient(p.potential)
    assert_allclose(p.energy_removed, g.energy(gx, gy), atol=2e-13)
    assert p.energy_removed >= 0
    assert_allclose(u, before[0], atol=0, rtol=0)
    assert_allclose(v, before[1], atol=0, rtol=0)


def test_poisson_matches_independent_dense_matrix():
    g = Grid(5, 4, lx=2, ly=3)
    matrix = np.zeros((20, 20))
    for j in range(g.ny):
        for i in range(g.nx):
            k = j * g.nx + i
            matrix[k, k] = -2 / g.dx**2 - 2 / g.dy**2
            for ii in ((i - 1) % g.nx, (i + 1) % g.nx):
                matrix[k, j * g.nx + ii] += 1 / g.dx**2
            for jj in ((j - 1) % g.ny, (j + 1) % g.ny):
                matrix[k, jj * g.nx + i] += 1 / g.dy**2
    rhs = np.random.default_rng(5).normal(size=g.shape)
    rhs -= rhs.mean()
    expected = np.linalg.solve(matrix + np.ones((20, 20)) / 20, rhs.ravel())
    assert_allclose(g.poisson(rhs).ravel(), expected, atol=2e-14)


def test_pressure_checkerboard_is_visible_on_mac():
    g = Grid(12, 10)
    j, i = np.indices(g.shape)
    p = (-1.0) ** (i + j)
    gx, gy = g.gradient(p)
    assert_allclose((np.roll(p, -1, 1) - np.roll(p, 1, 1)) / (2 * g.dx), 0)
    assert_allclose(np.abs(gx), 2 / g.dx)
    assert_allclose(np.abs(gy), 2 / g.dy)
    assert_allclose(g.poisson(g.laplacian(p)), p, atol=2e-14)


def test_discrete_curl_is_solenoidal_on_rectangular_grid():
    g = Grid(21, 13, lx=4.5, ly=2.7)
    psi = np.random.default_rng(2).normal(size=g.shape)
    u, v = g.curl(psi)
    assert_allclose(g.divergence(u, v), 0, atol=2e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nx": 2},
        {"ny": True},
        {"nx": 3.5},
        {"lx": 0},
        {"ly": -1},
        {"lx": np.inf},
        {"ly": np.nan},
        {"lx": 1e-300},
        {"ly": 1e300},
    ],
)
def test_invalid_grid(kwargs):
    args = {"nx": 8, "ny": 8} | kwargs
    with pytest.raises((ValueError, OverflowError)):
        Grid(**args)


@pytest.mark.parametrize(
    "value", [np.zeros((3, 4)), np.nan * np.ones((8, 8)), np.ones((8, 8), complex)]
)
def test_bad_fields(value):
    with pytest.raises(ValueError):
        Grid(8, 8).field(value)


def test_poisson_compatibility_and_coordinate_errors():
    g = Grid(8, 8)
    with pytest.raises(ValueError, match="zero mean"):
        g.poisson(np.ones(g.shape))
    with pytest.raises(ValueError, match="location"):
        g.coordinates("bad")


def test_native_boundary_checks_and_noncontiguous_input():
    g = _core.Grid(8, 8, 0.3, 0.2)
    with pytest.raises(ValueError):
        g.laplacian(np.ones(64))
    with pytest.raises(ValueError):
        g.gradient(np.full((8, 8), np.inf))
    values = np.arange(64.0).reshape(8, 8).T
    assert_allclose(g.laplacian(values), g.laplacian(np.ascontiguousarray(values)))
