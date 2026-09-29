"""Deterministic analytical studies and publication-sized diagnostic figure."""

import csv
import json
import platform
from pathlib import Path

import numpy as np

from .solver import Grid, project, stokes_step


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative(g, u, v, eu, ev):
    return float(np.sqrt(g.energy(u - eu, v - ev) / g.energy(eu, ev)))


def _vortex(g):
    xu, yu = g.coordinates("u")
    xv, yv = g.coordinates("v")
    return np.sin(xu) * np.cos(yu), -np.cos(xv) * np.sin(yv)


def run_study(output: str | Path) -> dict:
    """Write CSV, JSON, and PNG artifacts; return explicit numerical acceptance checks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spatial = []
    for n in (12, 24, 48, 96):
        g = Grid(n, n)
        u0, v0 = _vortex(g)
        u, v = u0.copy(), v0.copy()
        dt, nu, steps = 0.002, 0.25, 200
        for _ in range(steps):
            s = stokes_step(g, u, v, dt, nu)
            u, v = s.u, s.v
        exact = np.exp(-2 * nu * steps * dt)
        error = _relative(g, u, v, exact * u0, exact * v0)
        order = None if not spatial else float(np.log2(spatial[-1]["relative_error"] / error))
        spatial.append({"n": n, "h": g.dx, "dt": dt, "relative_error": error, "order": order})

    g = Grid(24, 18, lx=2 * np.pi, ly=3 * np.pi)
    x, y = g.coordinates("vertex")
    u0, v0 = g.curl(np.sin(2 * np.pi * x / g.lx) * np.sin(2 * np.pi * y / g.ly))
    temporal = []
    nu, end = 0.7, 1.2
    exact = np.exp(nu * g.eigenvalues[1, 1] * end)
    for steps in (6, 12, 24, 48):
        u, v = u0.copy(), v0.copy()
        for _ in range(steps):
            s = stokes_step(g, u, v, end / steps, nu)
            u, v = s.u, s.v
        error = _relative(g, u, v, exact * u0, exact * v0)
        order = None if not temporal else float(np.log2(temporal[-1]["relative_error"] / error))
        temporal.append(
            {"steps": steps, "dt": end / steps, "relative_error": error, "order": order}
        )

    # A known Helmholtz decomposition on a rectangular, anisotropic grid.
    g = Grid(48, 32, lx=2 * np.pi, ly=3 * np.pi)
    x, y = g.coordinates("vertex")
    su, sv = g.curl(np.sin(x) * np.sin(2 * y / 3))
    x, y = g.coordinates()
    potential = 0.35 * np.cos(2 * x) * np.cos(2 * y / 3)
    gx, gy = g.gradient(potential)
    wu, wv = su + gx + 0.2, sv + gy - 0.1
    p = project(g, wu, wv)
    before = g.divergence(wu, wv)
    after = g.divergence(p.u, p.v)
    p2 = project(g, p.u, p.v)
    uc, vc = 0.5 * (p.u + np.roll(p.u, -1, 1)), 0.5 * (p.v + np.roll(p.v, -1, 0))
    fields = [
        {
            "x": x[j, i],
            "y": y[j, i],
            "u_center": uc[j, i],
            "v_center": vc[j, i],
            "potential": p.potential[j, i],
            "divergence_before": before[j, i],
            "divergence_after": after[j, i],
        }
        for j in range(g.ny)
        for i in range(g.nx)
    ]

    # Nonzero initial flow, time-dependent vortical forcing and an independent
    # gradient force exercise both pressure and the work/dissipation ledger.
    u, v = su.copy(), sv.copy()
    initial_energy = g.energy(u, v)
    cumulative_work = cumulative_dissipation = 0.0
    energy = [
        {
            "time": 0.0,
            "energy": initial_energy,
            "work": 0.0,
            "dissipation": 0.0,
            "balance_defect": 0.0,
            "divergence_max": float(np.max(np.abs(g.divergence(u, v)))),
            "pressure_error_max": 0.0,
        }
    ]
    dt, nu = 0.02, 0.12
    for k in range(150):
        midpoint = (k + 0.5) * dt
        pressure = np.sin(midpoint) * potential
        fx, fy = g.gradient(pressure)
        force = np.cos(2 * midpoint) * su + fx, np.cos(2 * midpoint) * sv + fy
        s = stokes_step(g, u, v, dt, nu, force)
        u, v = s.u, s.v
        cumulative_work += dt * s.forcing_power
        cumulative_dissipation += dt * s.dissipation_power
        energy.append(
            {
                "time": (k + 1) * dt,
                "energy": s.energy,
                "work": cumulative_work,
                "dissipation": cumulative_dissipation,
                "balance_defect": s.energy
                - initial_energy
                - cumulative_work
                + cumulative_dissipation,
                "divergence_max": float(np.max(np.abs(g.divergence(u, v)))),
                "pressure_error_max": float(np.max(np.abs(s.pressure_midpoint - pressure))),
            }
        )

    theta = np.linspace(0, np.pi, 129)
    spectrum = [
        {
            "theta_over_pi": t / np.pi,
            "centered_gradient_times_h": abs(np.sin(t)),
            "mac_gradient_times_h": 2 * np.sin(t / 2),
        }
        for t in theta
    ]
    j, i = np.indices(g.shape)
    checker = (-1.0) ** (i + j)
    cx = (np.roll(checker, -1, 1) - np.roll(checker, 1, 1)) / (2 * g.dx)
    cgx, cgy = g.gradient(checker)
    metrics = {
        "spatial_finest_order": spatial[-1]["order"],
        "temporal_finest_order": temporal[-1]["order"],
        "projection_divergence_before_max": float(np.max(np.abs(before))),
        "projection_divergence_after_max": float(np.max(np.abs(after))),
        "projection_potential_error_max": float(np.max(np.abs(p.potential - potential))),
        "projection_idempotence_max": float(
            max(np.max(np.abs(p2.u - p.u)), np.max(np.abs(p2.v - p.v)))
        ),
        "projection_orthogonality_defect": p.orthogonality_defect,
        "projection_energy_removed": p.energy_removed,
        "projection_energy_identity_defect": abs(p.energy_removed - g.energy(gx, gy)),
        "forced_energy_balance_max": max(abs(row["balance_defect"]) for row in energy),
        "forced_divergence_max": max(row["divergence_max"] for row in energy),
        "forced_pressure_error_max": max(row["pressure_error_max"] for row in energy),
        "checkerboard_centered_x_gradient_max": float(np.max(np.abs(cx))),
        "checkerboard_mac_gradient_rms": float(np.sqrt(np.mean(cgx**2 + cgy**2))),
    }
    checks = {
        "space_second_order": 1.95 < metrics["spatial_finest_order"] < 2.05,
        "time_second_order": 1.95 < metrics["temporal_finest_order"] < 2.05,
        "projection_solenoidal": metrics["projection_divergence_after_max"] < 1e-11,
        "known_potential_recovered": metrics["projection_potential_error_max"] < 1e-12,
        "projection_idempotent": metrics["projection_idempotence_max"] < 1e-12,
        "projection_energy_identity": metrics["projection_energy_identity_defect"] < 1e-11,
        "forced_energy_balance": metrics["forced_energy_balance_max"] < 1e-10,
        "forced_incompressibility": metrics["forced_divergence_max"] < 1e-10,
        "forced_pressure_recovered": metrics["forced_pressure_error_max"] < 1e-12,
        "checkerboard_detected": metrics["checkerboard_centered_x_gradient_max"] == 0
        and metrics["checkerboard_mac_gradient_rms"] > 1,
    }
    report = {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "metrics": metrics,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    for name, rows in (
        ("spatial", spatial),
        ("temporal", temporal),
        ("fields", fields),
        ("energy", energy),
        ("pressure_spectrum", spectrum),
    ):
        _csv(output / f"{name}.csv", rows)
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot(output, g, x, y, uc, vc, p.potential, spatial, temporal, energy, spectrum, metrics)
    return report


def _plot(output, g, x, y, u, v, potential, spatial, temporal, energy, spectrum, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
        fig.suptitle(
            "StaggerLab | Pressure that enforces incompressibility", fontsize=19, weight="bold"
        )
        ax = axes[0, 0]
        mesh = ax.pcolormesh(x, y, np.hypot(u, v), shading="nearest", cmap="viridis")
        ax.quiver(x[::4, ::4], y[::4, ::4], u[::4, ::4], v[::4, ::4], color="white", alpha=0.85)
        fig.colorbar(mesh, ax=ax, label="Interpolated speed")
        ax.set(
            title="Projected flow on a rectangular grid", xlabel="x", ylabel="y", aspect="equal"
        )
        ax = axes[0, 1]
        mesh = ax.pcolormesh(x, y, potential, shading="nearest", cmap="RdBu_r")
        fig.colorbar(mesh, ax=ax, label="Velocity potential")
        ax.set(title="Recovered gradient component", xlabel="x", ylabel="y", aspect="equal")
        ax = axes[0, 2]
        for rows, key, label, marker in (
            (spatial, "h", "Space / continuum vortex", "o"),
            (temporal, "dt", "Time / discrete exponential", "s"),
        ):
            xx = np.array([row[key] for row in rows])
            yy = np.array([row["relative_error"] for row in rows])
            ax.loglog(xx / xx[0], yy / yy[0], marker + "-", label=label)
        xx = np.array([1, 0.125])
        ax.loglog(xx, xx**2, "k--", label="Second order")
        ax.set(
            title="Independent convergence studies",
            xlabel="h/h0 or dt/dt0",
            ylabel="Relative error / first error",
        )
        ax.legend(fontsize=8)
        ax.set_xticks([0.125, 0.25, 0.5, 1.0], labels=["1/8", "1/4", "1/2", "1"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2, which="both")
        ax = axes[1, 0]
        t = np.array([row["time"] for row in energy])
        e = np.array([row["energy"] for row in energy])
        ledger = e[0] + np.array([row["work"] - row["dissipation"] for row in energy])
        ax.plot(t, e, label="Computed kinetic energy", lw=2.5)
        ax.plot(t[::8], ledger[::8], "o", mfc="none", label="Initial + work - dissipation")
        ax.set(
            title="Forced-flow energy balance", xlabel="Time", ylabel="Energy (unit density)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax = axes[1, 1]
        ax.plot(
            [r["theta_over_pi"] for r in spectrum],
            [r["centered_gradient_times_h"] for r in spectrum],
            label="Collocated centered difference",
        )
        ax.plot(
            [r["theta_over_pi"] for r in spectrum],
            [r["mac_gradient_times_h"] for r in spectrum],
            label="MAC face gradient",
        )
        ax.axvline(1, color="0.5", linestyle=":")
        ax.set(
            title="Does pressure reach the velocity?",
            xlabel="Wavenumber / Nyquist",
            ylabel="Gradient magnitude × h",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax = axes[1, 2]
        ax.axis("off")
        details = (
            "NUMERICAL AUDIT\n\n"
            f"Spatial order                 {metrics['spatial_finest_order']:.4f}\n"
            f"Temporal order                {metrics['temporal_finest_order']:.4f}\n\n"
            f"Divergence before             {metrics['projection_divergence_before_max']:.2e}\n"
            f"Divergence after              {metrics['projection_divergence_after_max']:.2e}\n"
            f"Forced pressure error         {metrics['forced_pressure_error_max']:.2e}\n"
            f"Energy balance defect         {metrics['forced_energy_balance_max']:.2e}\n\n"
            "Periodic box · constant viscosity\n"
            "Linear Stokes flow · no advection\n"
            "C++ operators + Python FFT solves"
        )
        ax.text(
            0.04,
            0.95,
            details,
            va="top",
            family="monospace",
            fontsize=10,
            bbox={"facecolor": "#edf3f7", "edgecolor": "none", "pad": 14},
        )
        fig.savefig(output / "staggerlab_audit.png", dpi=160)
        plt.close(fig)
