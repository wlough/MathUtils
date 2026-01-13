// bind_shared_utils.cpp
#include "mathutils/bind/bind_shared_utils.hpp"
#include "mathutils/shared_utils.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// py::dict halfedge_to_dict(const HalfEdgeData& H) {
//     py::dict d;
//     d["xyz_coord_V"]   = H.xyz_coord_V;
//     d["V_cycle_E"]     = H.V_cycle_E;
//     d["V_cycle_F"]     = H.V_cycle_F;
//
//     d["h_out_V"]       = H.h_out_V;
//     d["h_directed_E"]  = H.h_directed_E;
//     d["h_right_F"]     = H.h_right_F;
//     d["h_negative_B"]  = H.h_negative_B;
//
//     d["v_origin_H"]    = H.v_origin_H;
//     d["e_undirected_H"]= H.e_undirected_H;
//     d["f_left_H"]      = H.f_left_H;
//
//     d["h_next_H"]      = H.h_next_H;
//     d["h_twin_H"]      = H.h_twin_H;
//     return d;
// }

// py::dict map_to_dict(const std::map<std::string, mathutils::Samplesi>& m) {
//     py::dict d;
//     for (const auto& pair : m) {
//         d[pair.first.c_str()] = pair.second;
//     }
//     return d;
// }

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
