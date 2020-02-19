#include <Eigen/Dense>
#include "SpherePlanning.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(SpherePlanning, m) {
    py::class_<SpherePlanning, SpherePlanning::Ptr>(m, "SpherePlanning")
        .def(py::init<>())
        .def("isValid", &SpherePlanning::isValid)
        .def("plan", &SpherePlanning::planOnce)
        .def("clear", &SpherePlanning::clear)
        .def("getPath", &SpherePlanning::getPath);
}