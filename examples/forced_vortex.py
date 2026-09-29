"""A reproducible pressure-driven equilibrium: gradient forces create no flow."""

import numpy as np

from staggerlab import Grid, stokes_step

grid = Grid(32, 24, lx=2 * np.pi, ly=3 * np.pi)
x, y = grid.coordinates()
pressure = 0.2 * np.sin(x) * np.cos(2 * y / 3)
zero = np.zeros(grid.shape)
step = stokes_step(grid, zero, zero, dt=0.05, viscosity=0.1, force=grid.gradient(pressure))
velocity_max = max(np.max(np.abs(step.u)), np.max(np.abs(step.v)))
pressure_error = np.max(np.abs(step.pressure_midpoint - pressure))
print(f"Velocity under a pure pressure gradient: {velocity_max:.3e}")
print(f"Zero-mean pressure recovery error: {pressure_error:.3e}")
if velocity_max > 1e-12 or pressure_error > 1e-12:
    raise RuntimeError("pressure equilibrium validation failed")
