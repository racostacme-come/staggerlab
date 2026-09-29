# Contributing

Use a focused branch for a numerical or documentation change. Keep formulas,
staggered field placement, and public return values documented. Include an
independent analytical or conservation-based check for numerical changes.
Do not import private course documents, exercise solutions, or datasets.

Before merging, run Ruff lint/format checks, clang-format, Python tests, native
CMake/CTest, a source/wheel build, an installed-wheel test, the example, and the
CLI study. The workflow in `.github/workflows/ci.yml` executes those checks on
Ubuntu and Windows with Python 3.11 and 3.14. CI is an additional environment
check; it does not replace scrutiny of equations and physical scope.

Regenerate `results/` intentionally with `staggerlab --output results`, inspect
the figure, and explain changes in numerical values. Keep test tolerances tied
to the relevant scale or explicitly bounded validation case. Never relax a
tolerance solely to make a failure disappear.

Use the configured Git identity and ordinary non-forced pushes. Preserve truthful
commits and verification records. Do not claim a review, platform, benchmark, or
test outcome that did not occur. This repository's initial history uses a
scaffold commit, a checked solver branch, and a validation/documentation branch.
