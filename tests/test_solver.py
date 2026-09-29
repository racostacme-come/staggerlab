import numpy as np
import pytest
from numpy.testing import assert_allclose

from staggerlab import Grid, project, stokes_step


def mode(g):
    x, y = g.coordinates("vertex")
    return g.curl(np.sin(2 * np.pi * x / g.lx) * np.sin(2 * np.pi * y / g.ly))


@pytest.mark.parametrize("viscosity", [0, 0.05, 3.0])
def test_uniform_flow_and_mean_forcing(viscosity):
    g = Grid(9, 7)
    u, v = np.full(g.shape, 2.0), np.full(g.shape, -1.0)
    f = np.full(g.shape, 0.3), np.full(g.shape, -0.7)
    s = stokes_step(g, u, v, 0.2, viscosity, f)
    assert_allclose(s.u, 2.06, atol=2e-15)
    assert_allclose(s.v, -1.14, atol=2e-15)
    assert_allclose(s.pressure_midpoint, 0, atol=2e-15)
    assert abs(s.energy_balance_defect) < 1e-12


def test_pure_gradient_force_changes_pressure_only():
    g = Grid(17, 12, lx=3.0, ly=4.1)
    x, y = g.coordinates()
    p = np.cos(2 * np.pi * x / g.lx) * np.sin(4 * np.pi * y / g.ly)
    zero = np.zeros(g.shape)
    s = stokes_step(g, zero, zero, 0.07, 0.1, g.gradient(p))
    assert_allclose(s.u, 0, atol=3e-15)
    assert_allclose(s.v, 0, atol=3e-15)
    assert_allclose(s.pressure_midpoint, p, atol=3e-15)


def test_full_discrete_momentum_and_energy_balance():
    g = Grid(19, 12, lx=4, ly=3)
    rng = np.random.default_rng(44)
    initial = project(g, *rng.normal(size=(2, *g.shape)))
    f = rng.normal(size=(2, *g.shape))
    dt, nu = 0.013, 0.07
    s = stokes_step(g, initial.u, initial.v, dt, nu, tuple(f))
    gx, gy = g.gradient(s.pressure_midpoint)
    for old, new, force, grad in zip(
        (initial.u, initial.v), (s.u, s.v), f, (gx, gy), strict=True
    ):
        residual = (new - old) / dt - nu * g.laplacian((old + new) / 2) + grad - force
        assert_allclose(residual, 0, atol=2e-12)
    assert_allclose(g.divergence(s.u, s.v), 0, atol=1e-12)
    assert abs(s.energy_balance_defect) < 2e-13
    assert s.dissipation_power >= 0


def test_unforced_energy_decreases_even_for_large_step():
    g = Grid(16, 14)
    p = project(g, *np.random.default_rng(82).normal(size=(2, *g.shape)))
    s = stokes_step(g, p.u, p.v, 3.0, 1.0)
    assert s.energy <= g.energy(p.u, p.v)
    assert abs(s.energy_balance_defect) < 1e-12


def test_temporal_order_against_semidiscrete_exponential():
    g = Grid(17, 13, lx=3, ly=4)
    u0, v0 = mode(g)
    rate = -0.3 * g.eigenvalues[1, 1]
    errors = []
    for steps in (8, 16, 32, 64):
        u, v = u0.copy(), v0.copy()
        for _ in range(steps):
            s = stokes_step(g, u, v, 0.7 / steps, 0.3)
            u, v = s.u, s.v
        errors.append(np.linalg.norm(u - np.exp(-rate * 0.7) * u0))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.99)


def test_time_varying_forcing_second_order():
    g = Grid(13, 9)
    u0, v0 = mode(g)
    nu, omega, end = 0.2, 2.3, 0.8
    rate = -nu * g.eigenvalues[1, 1]
    exact = (
        rate * np.cos(omega * end) + omega * np.sin(omega * end) - rate * np.exp(-rate * end)
    ) / (rate**2 + omega**2)
    errors = []
    for n in (8, 16, 32):
        u, v = np.zeros(g.shape), np.zeros(g.shape)
        for k in range(n):
            amp = np.cos(omega * (k + 0.5) * end / n)
            s = stokes_step(g, u, v, end / n, nu, (amp * u0, amp * v0))
            u, v = s.u, s.v
        errors.append(np.linalg.norm(u - exact * u0))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.98)


@pytest.mark.parametrize(
    "dt,nu",
    [(0, 0.1), (-1, 0.1), (np.nan, 0.1), (0.1, -1), (0.1, np.inf), (True, 1), (1e308, 1e308)],
)
def test_invalid_step_parameters(dt, nu):
    g = Grid(8, 8)
    z = np.zeros(g.shape)
    with pytest.raises(ValueError):
        stokes_step(g, z, z, dt, nu)


def test_divergent_initial_field_and_invalid_force_rejected():
    g = Grid(8, 8)
    z = np.zeros(g.shape)
    with pytest.raises(ValueError, match="divergence-free"):
        stokes_step(g, np.random.default_rng(9).normal(size=g.shape), z, 0.1, 0.1)
    with pytest.raises(ValueError, match="two face"):
        stokes_step(g, z, z, 0.1, 0.1, (z,))
