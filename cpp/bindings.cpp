#include "operators.hpp"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using staggerlab::Grid;

staggerlab::Field read(const Grid &g, const Array &a) {
    if (a.ndim() != 2 || a.shape(0) != static_cast<py::ssize_t>(g.ny) ||
        a.shape(1) != static_cast<py::ssize_t>(g.nx)) {
        throw std::invalid_argument("expected array shape (ny, nx)");
    }
    return {a.data(), a.data() + a.size()};
}

Array write(const Grid &g, const staggerlab::Field &a) {
    Array result({static_cast<py::ssize_t>(g.ny), static_cast<py::ssize_t>(g.nx)});
    std::copy(a.begin(), a.end(), result.mutable_data());
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Checked C++17 periodic MAC difference operators";
    py::class_<Grid>(m, "Grid")
        .def(py::init<std::size_t, std::size_t, double, double>())
        .def("divergence",
             [](const Grid &g, const Array &u, const Array &v) {
                 return write(g, staggerlab::divergence(g, read(g, u), read(g, v)));
             })
        .def("gradient",
             [](const Grid &g, const Array &p) {
                 const auto pair = staggerlab::gradient(g, read(g, p));
                 return py::make_tuple(write(g, pair.first), write(g, pair.second));
             })
        .def("laplacian", [](const Grid &g, const Array &a) {
            return write(g, staggerlab::laplacian(g, read(g, a)));
        });
}
