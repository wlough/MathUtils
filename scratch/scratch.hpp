// bind_mesh.cpp
#include "mathutils/bind/bind_mesh.hpp"
#include "mathutils/bind/matrix_type_caster.hpp"
#include "mathutils/mesh/mesh.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_mesh(py::module_ &m) {
  using mathutils::mesh::HalfPlexMesh;
  using mathutils::mesh::SimplicialComplexBase;

  // Register base type (not meant for user-facing API)
  py::class_<SimplicialComplexBase>(m, "_SimplicialComplexBase");

  // Public class
  py::class_<HalfPlexMesh, SimplicialComplexBase>(m, "HalfPlexMesh")
      .def(py::init<>())

      // storage accessors (refs)
      .def("h_out_V", &HalfPlexMesh::h_out_V,
           py::return_value_policy::reference_internal)
      .def("h_directed_E", &HalfPlexMesh::h_directed_E,
           py::return_value_policy::reference_internal)
      .def("h_right_F", &HalfPlexMesh::h_right_F,
           py::return_value_policy::reference_internal)
      .def("h_above_C", &HalfPlexMesh::h_above_C,
           py::return_value_policy::reference_internal)
      .def("h_negative_B", &HalfPlexMesh::h_negative_B,
           py::return_value_policy::reference_internal)

      .def("v_origin_H", &HalfPlexMesh::v_origin_H,
           py::return_value_policy::reference_internal)
      .def("e_undirected_H", &HalfPlexMesh::e_undirected_H,
           py::return_value_policy::reference_internal)
      .def("f_left_H", &HalfPlexMesh::f_left_H,
           py::return_value_policy::reference_internal)
      .def("c_below_H", &HalfPlexMesh::c_below_H,
           py::return_value_policy::reference_internal)

      .def("h_next_H", &HalfPlexMesh::h_next_H,
           py::return_value_policy::reference_internal)
      .def("h_twin_H", &HalfPlexMesh::h_twin_H,
           py::return_value_policy::reference_internal)
      .def("h_flip_H", &HalfPlexMesh::h_flip_H,
           py::return_value_policy::reference_internal)

      // scalar queries
      .def("h_out_v", &HalfPlexMesh::h_out_v, py::arg("v"))
      .def("h_directed_e", &HalfPlexMesh::h_directed_e, py::arg("e"))
      .def("h_right_f", &HalfPlexMesh::h_right_f, py::arg("f"))
      .def("h_above_c", &HalfPlexMesh::h_above_c, py::arg("c"))
      .def("h_negative_b", &HalfPlexMesh::h_negative_b, py::arg("b"))

      .def("v_origin_h", &HalfPlexMesh::v_origin_h, py::arg("h"))
      .def("e_undirected_h", &HalfPlexMesh::e_undirected_h, py::arg("h"))
      .def("f_left_h", &HalfPlexMesh::f_left_h, py::arg("h"))
      .def("c_below_h", &HalfPlexMesh::c_below_h, py::arg("h"))

      .def("h_next_h", &HalfPlexMesh::h_next_h, py::arg("h"))
      .def("h_twin_h", &HalfPlexMesh::h_twin_h, py::arg("h"))
      .def("h_flip_h", &HalfPlexMesh::h_flip_h, py::arg("h"))

      .def("b_ghost_f", &HalfPlexMesh::b_ghost_f, py::arg("f"))
      .def("b_ghost_c", &HalfPlexMesh::b_ghost_c, py::arg("c"))
      .def("h_in_v", &HalfPlexMesh::h_in_v, py::arg("v"))
      .def("v_head_h", &HalfPlexMesh::v_head_h, py::arg("h"))
      .def("h_prev_h", &HalfPlexMesh::h_prev_h, py::arg("h"))

      // topology / counts
      .def("num_vertices", &HalfPlexMesh::num_vertices)
      .def("num_edges", &HalfPlexMesh::num_edges)
      .def("num_faces", &HalfPlexMesh::num_faces)
      .def("num_cells", &HalfPlexMesh::num_cells)
      .def("num_half_edges", &HalfPlexMesh::num_half_edges)
      .def("euler_characteristic", &HalfPlexMesh::euler_characteristic)
      .def("num_boundaries", &HalfPlexMesh::num_boundaries)
      .def("genus", &HalfPlexMesh::genus)

      .def("to_mesh_samples", &HalfPlexMesh::to_mesh_samples);
}
