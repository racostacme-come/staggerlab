#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace staggerlab {
using Field = std::vector<double>;

struct Grid {
    std::size_t nx, ny;
    double dx, dy;
    Grid(std::size_t nx_, std::size_t ny_, double dx_, double dy_)
        : nx(nx_), ny(ny_), dx(dx_), dy(dy_) {
        if (nx < 3 || ny < 3 || nx > std::numeric_limits<std::size_t>::max() / ny ||
            !std::isfinite(dx) || !std::isfinite(dy) || dx <= 0 || dy <= 0 ||
            !std::isfinite(1.0 / (dx * dx) + 1.0 / (dy * dy)) || dx * dx == 0 || dy * dy == 0 ||
            !std::isfinite(dx * dx + dy * dy)) {
            throw std::invalid_argument("invalid periodic grid dimensions or spacing");
        }
    }
    std::size_t index(std::size_t j, std::size_t i) const { return j * nx + i; }
    void check(const Field &a) const {
        if (a.size() != nx * ny) {
            throw std::invalid_argument("field size differs from grid");
        }
        for (const double value : a) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("field must be finite");
            }
        }
    }
};

inline Field divergence(const Grid &g, const Field &u, const Field &v) {
    g.check(u);
    g.check(v);
    Field result(u.size());
    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const auto k = g.index(j, i);
            result[k] = (u[g.index(j, (i + 1) % g.nx)] - u[k]) / g.dx +
                        (v[g.index((j + 1) % g.ny, i)] - v[k]) / g.dy;
        }
    }
    return result;
}

inline std::pair<Field, Field> gradient(const Grid &g, const Field &p) {
    g.check(p);
    Field x(p.size()), y(p.size());
    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const auto k = g.index(j, i);
            x[k] = (p[k] - p[g.index(j, (i + g.nx - 1) % g.nx)]) / g.dx;
            y[k] = (p[k] - p[g.index((j + g.ny - 1) % g.ny, i)]) / g.dy;
        }
    }
    return {std::move(x), std::move(y)};
}

inline Field laplacian(const Grid &g, const Field &a) {
    g.check(a);
    Field result(a.size());
    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const auto k = g.index(j, i);
            result[k] = (a[g.index(j, (i + 1) % g.nx)] - 2 * a[k] +
                         a[g.index(j, (i + g.nx - 1) % g.nx)]) /
                            (g.dx * g.dx) +
                        (a[g.index((j + 1) % g.ny, i)] - 2 * a[k] +
                         a[g.index((j + g.ny - 1) % g.ny, i)]) /
                            (g.dy * g.dy);
        }
    }
    return result;
}
} // namespace staggerlab
