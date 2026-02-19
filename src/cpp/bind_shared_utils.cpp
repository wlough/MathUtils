// bind_shared_utils.cpp
#include "mathutils/bind/bind_shared_utils.hpp"
#include "mathutils/shared_utils.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_shared_utils(py::module_ &m) {
  // Basic utility functions (in main module)
  m.def("sign", &mathutils::sign<double>, "sign function (double)",
        py::arg("x"));
  m.def("sign", &mathutils::sign<int>, "sign function (int)", py::arg("x"));
  m.def("thetaphi_from_xyz", &mathutils::thetaphi_from_xyz,
        "Convert Cartesian coordinates (x, y, z) to unit sphere coordinates "
        "(theta, phi)",
        py::arg("xyz"));
  m.def("xyz_from_rthetaphi", &mathutils::xyz_from_rthetaphi,
        "Convert spherical coordinates (r, theta, phi) to Cartesian "
        "coordinates (x, y, z)",
        py::arg("rthetaphi"));
  m.def("rthetaphi_from_xyz", &mathutils::rthetaphi_from_xyz,
        "Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, "
        "theta, phi)",
        py::arg("xyz"));
}
