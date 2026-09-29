"""Periodic, constant-viscosity Stokes flow; all fields have shape (ny, nx)."""

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

Field = NDArray[np.float64]


def _scalar(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, (bool, complex)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


@dataclass(frozen=True)
class Grid:
    """Uniform periodic box; no duplicate endpoints or ghost cells.

    p[j,i]: ((i+.5)dx, (j+.5)dy); u[j,i]: (i dx, (j+.5)dy);
    v[j,i]: ((i+.5)dx, j dy). The velocity components live on faces.
    """

    nx: int
    ny: int
    lx: float = 2 * np.pi
    ly: float = 2 * np.pi

    def __post_init__(self):
        for name in ("nx", "ny"):
            n = getattr(self, name)
            if isinstance(n, bool) or not isinstance(n, Integral) or n < 3:
                raise ValueError(f"{name} must be an integer >= 3")
            object.__setattr__(self, name, int(n))
        object.__setattr__(self, "lx", _scalar(self.lx, "lx"))
        object.__setattr__(self, "ly", _scalar(self.ly, "ly"))
        # Construction checks spacing representability without allocating arrays.
        object.__setattr__(self, "_native", _core.Grid(self.nx, self.ny, self.dx, self.dy))

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def field(self, value: ArrayLike) -> Field:
        """Validate and normalize a real finite grid field; may share input memory."""
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError("fields must be real")
        a = np.asarray(raw, dtype=np.float64, order="C")
        if a.shape != self.shape or not np.isfinite(a).all():
            raise ValueError(f"expected finite field of shape {self.shape}")
        return a

    def coordinates(self, location: str = "p") -> tuple[Field, Field]:
        """Return x,y arrays at p, u, v, or vertex locations."""
        offsets = {"p": (0.5, 0.5), "u": (0.0, 0.5), "v": (0.5, 0.0), "vertex": (0, 0)}
        if location not in offsets:
            raise ValueError("location must be p, u, v, or vertex")
        ox, oy = offsets[location]
        return np.meshgrid(
            (np.arange(self.nx) + ox) * self.dx, (np.arange(self.ny) + oy) * self.dy
        )

    def divergence(self, u: ArrayLike, v: ArrayLike) -> Field:
        return self._native.divergence(self.field(u), self.field(v))

    def gradient(self, p: ArrayLike) -> tuple[Field, Field]:
        return self._native.gradient(self.field(p))

    def laplacian(self, a: ArrayLike) -> Field:
        return self._native.laplacian(self.field(a))

    def curl(self, psi: ArrayLike) -> tuple[Field, Field]:
        """Map a vertex streamfunction to discretely divergence-free face velocities."""
        psi = self.field(psi)
        return (np.roll(psi, -1, axis=0) - psi) / self.dy, -(
            np.roll(psi, -1, axis=1) - psi
        ) / self.dx

    @cached_property
    def eigenvalues(self) -> Field:
        """Nonpositive eigenvalues of DG, in NumPy's full FFT ordering."""
        ex = -4 * (np.sin(np.pi * np.fft.fftfreq(self.nx)) / self.dx) ** 2
        ey = -4 * (np.sin(np.pi * np.fft.fftfreq(self.ny)) / self.dy) ** 2
        values = ey[:, None] + ex[None, :]
        values.setflags(write=False)
        return values

    def poisson(self, rhs: ArrayLike) -> Field:
        """Solve DG p = rhs with mean(p)=0; reject incompatible nonzero means.

        A mean up to 1e-12*max(1,max(abs(rhs))) is treated as roundoff and removed.
        """
        rhs = self.field(rhs)
        mean = float(np.mean(rhs))
        if abs(mean) > 1e-12 * max(1.0, float(np.max(np.abs(rhs)))):
            raise ValueError("periodic Poisson right-hand side must have zero mean")
        spectrum = np.fft.fft2(rhs - mean)
        result = np.zeros_like(spectrum)
        np.divide(spectrum, self.eigenvalues, out=result, where=self.eigenvalues != 0)
        return np.fft.ifft2(result).real

    def inner(self, a: ArrayLike, b: ArrayLike) -> float:
        """Discrete integral of a*b over the periodic box."""
        return float(self.dx * self.dy * np.sum(self.field(a) * self.field(b)))

    def energy(self, u: ArrayLike, v: ArrayLike) -> float:
        """Kinetic energy for unit density, using staggered face quadrature."""
        return 0.5 * (self.inner(u, u) + self.inner(v, v))


@dataclass(frozen=True)
class Projection:
    u: Field
    v: Field
    potential: Field
    energy_removed: float
    orthogonality_defect: float


def project(grid: Grid, u: ArrayLike, v: ArrayLike) -> Projection:
    """Orthogonal Helmholtz projection P=I-G(DG)^dagger D.

    The returned potential is a velocity potential, not a physical pressure.
    Inputs are never modified.
    """
    u, v = grid.field(u), grid.field(v)
    potential = grid.poisson(grid.divergence(u, v))
    gx, gy = grid.gradient(potential)
    pu, pv = u - gx, v - gy
    defect = grid.inner(pu, gx) + grid.inner(pv, gy)
    return Projection(pu, pv, potential, grid.energy(u, v) - grid.energy(pu, pv), defect)


@dataclass(frozen=True)
class Step:
    u: Field
    v: Field
    pressure_midpoint: Field
    energy: float
    dissipation_power: float
    forcing_power: float
    energy_balance_defect: float


def stokes_step(
    grid: Grid,
    u: ArrayLike,
    v: ArrayLike,
    dt: float,
    viscosity: float,
    force: tuple[ArrayLike, ArrayLike] | None = None,
) -> Step:
    """One Crank-Nicolson step with forcing supplied at the time midpoint.

    Solves du/dt = -Gp + nu*L*u + f, D*u=0, with constant viscosity.
    The initial velocity must be solenoidal; call project explicitly if needed.
    Pressure is the zero-mean midpoint pressure from L*p=D*f.
    """
    dt = _scalar(dt, "dt")
    viscosity = _scalar(viscosity, "viscosity", zero=True)
    u, v = grid.field(u), grid.field(v)
    scale = max(1.0, float(np.max(np.abs(u))) / grid.dx + float(np.max(np.abs(v))) / grid.dy)
    if np.max(np.abs(grid.divergence(u, v))) > 1e-10 * scale:
        raise ValueError("initial velocity must be discretely divergence-free; use project")
    if force is None:
        fu, fv = np.zeros(grid.shape), np.zeros(grid.shape)
    else:
        if len(force) != 2:
            raise ValueError("force must contain two face fields")
        fu, fv = grid.field(force[0]), grid.field(force[1])
    pf = project(grid, fu, fv)
    with np.errstate(over="ignore", invalid="ignore"):
        a = 0.5 * dt * viscosity * grid.eigenvalues
    if not np.isfinite(a).all():
        raise ValueError("dt*viscosity*L exceeds floating-point range")
    multiplier = (1 + a) / (1 - a)

    def advance(old, f):
        new = np.fft.ifft2(multiplier * np.fft.fft2(old) + dt * np.fft.fft2(f) / (1 - a)).real
        if not np.isfinite(new).all():
            raise ValueError("step produced nonfinite velocities")
        return new

    un, vn = advance(u, pf.u), advance(v, pf.v)
    um, vm = (u + un) / 2, (v + vn) / 2
    energy = grid.energy(un, vn)
    dissipation = -viscosity * (
        grid.inner(um, grid.laplacian(um)) + grid.inner(vm, grid.laplacian(vm))
    )
    work = grid.inner(um, fu) + grid.inner(vm, fv)
    defect = energy - grid.energy(u, v) + dt * (dissipation - work)
    return Step(un, vn, pf.potential, energy, dissipation, work, defect)
