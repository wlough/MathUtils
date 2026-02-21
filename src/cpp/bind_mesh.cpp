// bind_mesh.cpp
#include "mathutils/bind/bind_mesh.hpp"
#include "mathutils/bind/matrix_type_caster.hpp"
#include "mathutils/bind/numpy_view_span.hpp"
#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_mesh(py::module_ &m) {

  m.doc() = "Mesh utilities";

  m.def("find_halfedge_index_of_twin",
        &mathutils::mesh::find_halfedge_index_of_twin,
        "Find the index of the twin half-edge", py::arg("H"), py::arg("h"),
        R"doc(
Find the index ht of H[ht]=[j, i] in H, where H[h]=[i, j]. Return -1 if not found.
)doc");

  m.def("tri_vertex_cycles_to_half_edge_samples",
        &mathutils::mesh::tri_vertex_cycles_to_half_edge_samples,
        "Convert triangle vertex cycles to half-edge samples",
        py::arg("V_cycle_F"),
        R"doc(
Convert triangle vertex cycles to half-edge samples.
)doc");

  m.def("load_vf_samples_from_ply",
        &mathutils::mesh::io::load_vf_samples_from_ply,
        "Load vertex and face samples from a PLY file", py::arg("filepath"),
        py::arg("preload_into_memory") = true, py::arg("verbose") = false,
        R"doc(
Load vertex and face samples from a PLY file.

Args:
    filepath: Path to the PLY file
    preload_into_memory: Whether to preload the file into memory (default: True)
    verbose: Whether to print verbose output (default: False)

Returns:
    A tuple of (vertices, faces) where:
    - vertices is an Nx3 array of vertex coordinates
    - faces is an Mx3 array of triangle vertex indices
)doc");

  m.def("write_vf_samples_to_ply",
        &mathutils::mesh::io::write_vf_samples_to_ply,
        "Write vertex and face samples to a PLY file", py::arg("xyz_coord_V"),
        py::arg("V_cycle_F"), py::arg("ply_path"), py::arg("use_binary") = true,
        R"doc(
  Write vertex and face samples to a PLY file.

  Args:
      xyz_coord_V: Nx3 array of vertex coordinates
      V_cycle_F: Mx3 array of triangle vertex indices
      ply_path: Output PLY file path
      use_binary: Whether to write binary format (default: True)
  )doc");

  m.def("load_mesh_samples_from_ply",
        &mathutils::mesh::io::load_mesh_samples_from_ply,
        "Load mesh samples from a PLY file", py::arg("filepath"),
        py::arg("preload_into_memory") = true, py::arg("verbose") = false,
        R"doc(
Load mesh samples from a PLY file.
Args:
    filepath: Path to the PLY file
    preload_into_memory: Whether to preload the file into memory (default: True)
    verbose: Whether to print verbose output (default: False)
Returns:
    A dict of mesh samples with keys as sample names and values as ndarrays.
)doc");

  m.def("write_mesh_samples_to_ply",
        &mathutils::mesh::io::write_mesh_samples_to_ply,
        "Write mesh samples to a PLY file", py::arg("mesh_samples"),
        py::arg("ply_path"), py::arg("use_binary") = true,
        R"doc(
Write mesh samples to a PLY file.
Args:
    mesh_samples: A dict of mesh samples with keys as sample names and values as ndarrays.
    ply_path: Output PLY file path
    use_binary: Whether to write binary format (default: True)
)doc");

  m.def("save_mesh_samples", &mathutils::mesh::io::save_mesh_samples,
        "Write mesh samples to a PLY file", py::arg("mesh_samples"),
        py::arg("ply_path"), py::arg("use_binary") = true,
        py::arg("ply_property_convention") = "MathUtils");

  m.def("load_mesh_samples", &mathutils::mesh::io::load_mesh_samples,
        "Load mesh samples from a PLY file", py::arg("filepath"),
        py::arg("preload_into_memory") = true, py::arg("verbose") = false,
        py::arg("ply_property_convention") = "MathUtils");

  //////////////////////////////////////////////
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // HalfEdgeTopology
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  using mathutils::mesh::HalfEdgeMesh;
  using mathutils::mesh::HalfEdgeTopology;
  using mathutils::mesh::Index;
  using mathutils::mesh::Real;

  // using mathutils::mesh::SimplicialComplexBase;

  // Register base type (not meant for user-facing API)
  // py::class_<SimplicialComplexBase>(m, "_SimplicialComplexBase");

  // Public class
  py::class_<HalfEdgeTopology>(m, "HalfEdgeTopology")
      .def(py::init<>())

      // storage accessors (refs)
      // .def("X_ambient_V", &HalfEdgeTopology::X_ambient_V,
      //      py::return_value_policy::reference_internal)
      // .def("V_cycle_E", &HalfEdgeTopology::V_cycle_E,
      //      py::return_value_policy::reference_internal)
      // .def("V_cycle_F", &HalfEdgeTopology::V_cycle_F,
      //      py::return_value_policy::reference_internal)
      // .def("V_cycle_C", &HalfEdgeTopology::V_cycle_C,
      //      py::return_value_policy::reference_internal)

      .def("h_out_V", &HalfEdgeTopology::h_out_V,
           py::return_value_policy::reference_internal)
      .def("h_directed_E", &HalfEdgeTopology::h_directed_E,
           py::return_value_policy::reference_internal)
      .def("h_right_F", &HalfEdgeTopology::h_right_F,
           py::return_value_policy::reference_internal)
      .def("h_negative_B", &HalfEdgeTopology::h_negative_B,
           py::return_value_policy::reference_internal)

      .def("v_origin_H", &HalfEdgeTopology::v_origin_H,
           py::return_value_policy::reference_internal)
      .def("e_undirected_H", &HalfEdgeTopology::e_undirected_H,
           py::return_value_policy::reference_internal)
      .def("f_left_H", &HalfEdgeTopology::f_left_H,
           py::return_value_policy::reference_internal)

      .def("h_next_H", &HalfEdgeTopology::h_next_H,
           py::return_value_policy::reference_internal)
      .def("h_twin_H", &HalfEdgeTopology::h_twin_H,
           py::return_value_policy::reference_internal)

      // scalar queries
      .def("h_out_v", &HalfEdgeTopology::h_out_v, py::arg("v"))
      .def("h_directed_e", &HalfEdgeTopology::h_directed_e, py::arg("e"))
      .def("h_right_f", &HalfEdgeTopology::h_right_f, py::arg("f"))
      .def("h_negative_b", &HalfEdgeTopology::h_negative_b, py::arg("b"))

      .def("v_origin_h", &HalfEdgeTopology::v_origin_h, py::arg("h"))
      .def("e_undirected_h", &HalfEdgeTopology::e_undirected_h, py::arg("h"))
      .def("f_left_h", &HalfEdgeTopology::f_left_h, py::arg("h"))

      .def("h_next_h", &HalfEdgeTopology::h_next_h, py::arg("h"))
      .def("h_twin_h", &HalfEdgeTopology::h_twin_h, py::arg("h"))

      .def("b_ghost_f", &HalfEdgeTopology::b_ghost_f, py::arg("f"))
      .def("h_in_v", &HalfEdgeTopology::h_in_v, py::arg("v"))
      .def("v_head_h", &HalfEdgeTopology::v_head_h, py::arg("h"))
      .def("h_prev_h", &HalfEdgeTopology::h_prev_h, py::arg("h"))

      // topology / counts
      .def("num_vertices", &HalfEdgeTopology::num_vertices)
      .def("num_edges", &HalfEdgeTopology::num_edges)
      .def("num_faces", &HalfEdgeTopology::num_faces)
      .def("num_half_edges", &HalfEdgeTopology::num_half_edges)
      .def("euler_characteristic", &HalfEdgeTopology::euler_characteristic)
      .def("num_boundaries", &HalfEdgeTopology::num_boundaries)
      .def("genus", &HalfEdgeTopology::genus)

      .def("to_mesh_samples", &HalfEdgeTopology::to_mesh_samples)
      .def("from_mesh_samples", &HalfEdgeTopology::from_mesh_samples);

  py::class_<HalfEdgeMesh>(m, "HalfEdgeMesh")
      .def(py::init<>())

      .def_property_readonly(
          "topo",
          [](HalfEdgeMesh &self) -> HalfEdgeTopology & { return self.topo; },
          py::return_value_policy::reference_internal)

      .def_property_readonly(
          "attrs",
          [](HalfEdgeMesh &self) -> mathutils::mesh::MeshSamples & {
            return self.attrs;
          },
          py::return_value_policy::reference_internal)

      .def("X_ambient_V", &HalfEdgeMesh::X_ambient_V,
           py::return_value_policy::reference_internal)

      .def(
          "X_ambient_v",
          [](HalfEdgeMesh &self, Index v) {
            std::span<Real> s = self.X_ambient_v(v);
            return numpy_view_span(s, py::cast(&self));
          },
          py::arg("v"),
          "Return a writable NumPy view of the vertex position row (shape "
          "(3,), etc.).")
      .def("to_mesh_samples", &HalfEdgeMesh::to_mesh_samples)
      .def("from_mesh_samples", &HalfEdgeMesh::from_mesh_samples)
      .def("load_ply", &HalfEdgeMesh::load_ply,
           "Load mesh samples from a PLY file", py::arg("filepath"),
           py::arg("preload_into_memory") = true, py::arg("verbose") = false,
           py::arg("ply_property_convention") = "MathUtils");
}
