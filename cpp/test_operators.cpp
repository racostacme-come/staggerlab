#include "operators.hpp"
#include <algorithm>
#include <iostream>

int main() {
    using namespace staggerlab;
    const Grid g(13, 9, 0.17, 0.29);
    Field p(g.nx * g.ny), u(p.size()), v(p.size());
    for (std::size_t k = 0; k < p.size(); ++k) {
        p[k] = std::sin(0.19 * static_cast<double>(k));
        u[k] = std::cos(0.31 * static_cast<double>(k));
        v[k] = std::sin(0.11 * static_cast<double>(k));
    }
    const auto gp = gradient(g, p);
    const auto dg = divergence(g, gp.first, gp.second);
    const auto lp = laplacian(g, p);
    const auto du = divergence(g, u, v);
    double adjoint = 0, conservation = 0, mismatch = 0;
    for (std::size_t k = 0; k < p.size(); ++k) {
        adjoint += p[k] * du[k] + u[k] * gp.first[k] + v[k] * gp.second[k];
        conservation += du[k];
        mismatch = std::max(mismatch, std::abs(dg[k] - lp[k]));
    }
    if (std::abs(adjoint) > 1e-11 || std::abs(conservation) > 1e-11 || mismatch > 1e-11) {
        std::cerr << "compatible operator identities failed\n";
        return 1;
    }
    bool rejected = false;
    try {
        laplacian(g, Field(2));
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        return 2;
    }
    rejected = false;
    try {
        Grid invalid(1, 9, 0.1, 0.1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    return rejected ? 0 : 3;
}
