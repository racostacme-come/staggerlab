# Validation record

Local execution: 2026-09-29, Windows, CPython 3.14.5, NumPy 2.5.3,
Matplotlib 3.11.2, MSVC 19.44.35219.0, CMake with Visual Studio 17 2022.
The package has compiled C++17 kernels; coverage percentages below measure
Python statements only, excluding compiled code.

## Independent numerical checks

1. **Spatial convergence.** On `[0,2pi)^2`, sample
   `u=sin(x)cos(y)`, `v=-cos(x)sin(y)` at their respective faces. Equal x/y
   spacing makes this sampled field discretely solenoidal. With `nu=0.25`, the
   continuum decay at `T=0.4` is `exp(-2 nu T)`. Use 12, 24, 48, 96 cells per
   direction and 200 steps of `dt=0.002`. Relative combined velocity L2 errors
   are 4.53796e-3, 1.14034e-3, 2.85440e-4, 7.13704e-5; last order 1.99979.
   The fixed timestep leaves a small temporal error; this is observed spatial
   convergence over the stated range, not an arbitrarily fine-grid claim.
2. **Temporal convergence.** A discrete curl mode on a 24×18 grid over
   `[0,2pi)×[0,3pi)` evolves with exact semidiscrete factor
   `exp(nu*lambda[1,1]*T)`. Use `nu=0.7`, `T=1.2`, and 6, 12, 24, 48 steps.
   Relative errors are 4.06428e-3, 1.01299e-3, 2.53057e-4, 6.32524e-5;
   last order 2.00027. A separate unit test integrates cosine forcing against
   the closed-form scalar modal ODE and checks second order.
3. **Known Helmholtz decomposition.** A 48×32 rectangular grid combines a
   discrete curl, constant velocity means `(0.2,-0.1)`, and the gradient of
   `phi=0.35 cos(2x) cos(2y/3)`. Check recovered potential, projection
   idempotence, divergence, orthogonality, and removed energy.
4. **Forced Stokes flow.** Start from the discrete curl on that rectangular
   grid; force it with `cos(2t)*u0 + G(sin(t)*phi)`. Use `nu=0.12`, `dt=0.02`,
   150 steps. Evaluate forcing at each midpoint. The pressure is
   `sin(t_mid)*phi`. Accumulate supplied work and viscous dissipation; their
   difference agrees with the kinetic-energy change within 1.20e-13.
   Unit tests also check the full two-component discrete momentum residual
   for a random solenoidal initial field and random forcing.
5. **Pressure checkerboard.** On an even grid, `(-1)^(i+j)` has exactly zero
   centered collocated x gradient; its MAC gradient RMS is 16.71994 on the
   study grid. A test recovers this pressure from its discrete Laplacian.
6. **Independent Poisson solve.** Compare the FFT solution on a 5×4 grid with
   a separately assembled dense five-point matrix, adding the constant-mode
   projector to fix the gauge. Other tests cover adjointness, conservation,
   noncontiguous arrays, odd/even and minimal grids, pure-gradient force,
   uniform velocity/forcing, large-step energy stability, and bad inputs.

## Acceptance criteria

The CLI records ten boolean checks, all required for exit 0:

| Check | Threshold |
| --- | --- |
| Finest spatial and temporal orders | each in (1.95, 2.05) |
| Projected maximum divergence | < 1e-11 |
| Known potential recovery | max error < 1e-12 |
| Projection idempotence | max change < 1e-12 |
| Projection energy identity | absolute defect < 1e-11 |
| Cumulative forced energy ledger | max absolute defect < 1e-10 |
| Forced velocity divergence | maximum < 1e-10 |
| Forced pressure recovery | max error < 1e-12 |
| Checkerboard diagnosis | centered x gradient = 0; MAC gradient RMS > 1 |

These are case-specific absolute tolerances for nondimensional manufactured
problems. They are not general relative error bounds or certified enclosures.

## Verification performed during development

- Solver branch: 38 Python tests and the standalone native CTest passed before
  merging. The native checks use explicit failure returns and remain active
  in Release builds. Compiler warnings are errors (`/W4 /WX` or
  `-Wall -Wextra -Wpedantic -Werror`).
- Validation stage: 41 Python tests passed, with 272/274 Python statements
  covered after the final plot-label edit (99.27%). The uncovered statements
  are the defensive nonfinite-output exception and module CLI entry point;
  the installed CLI is also exercised outside coverage.
- Ruff lint and format checks, clang-format, example, CLI acceptance checks,
  and dependency consistency passed locally. Initial style failures in newly
  written files were corrected before committing the validation branch.
- The generated figure was inspected visually. Crowded logarithmic x-axis
  tick labels were replaced by the four actual refinement ratios.
- Isolated source and wheel builds succeeded, including compiling the wheel
  from the source archive. Both archives were checked for the required package,
  native source, tests, and documentation, and absence of environments/build trees.
  The wheel contains the compiled extension. MSBuild emitted temporary-directory
  and output-path warnings during isolated packaging; compilation and linking
  completed successfully. The standalone native build passed without those
  packaging-directory warnings.
- In a separate fresh environment, the installed wheel passed all 41 tests,
  the example, the ten-check CLI study, and `pip check`. Its import location
  was verified inside `site-packages`, independently of the editable checkout.

The repository's Actions page records actual CI results for each commit; a
workflow definition alone is not evidence that a particular run passed.
Final CI outcomes are recorded with the completed project in the series index
and automation memory. No performance or cross-platform
bitwise reproducibility claim is made.

## Artifact interpretation

`spatial.csv` and `temporal.csv` retain raw errors; first-row orders are empty
because no preceding refinement exists. The figure normalizes each curve by its
first error only for visual comparison. `fields.csv` uses cell-centered
interpolated velocities, original and projected cell divergence, and the
projection potential. `energy.csv` includes an initial row at time zero,
cumulative work/dissipation, divergence, and midpoint-pressure errors on later
rows. `pressure_spectrum.csv` is the analytic one-dimensional gradient-symbol
magnitude comparison, not a fluid energy spectrum. `validation.json` stores
unrounded metrics and explicit checks. PNG rendering may vary with fonts and
Matplotlib versions even when numerical checks agree.
