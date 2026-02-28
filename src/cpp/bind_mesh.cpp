// bind_mesh.cpp
#include "mathutils/bind/bind_mesh.hpp"
#include "mathutils/bind/matrix_type_caster.hpp"
#include "mathutils/bind/matrix_view.hpp"
#include "mathutils/bind/span_view.hpp"
#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/mesh_convert_funs.hpp"
#include "mathutils/mesh/mesh_plyio.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(mathutils::mesh::MeshSamples);

namespace py = pybind11;

void bind_mesh(py::module_ &m) {

  m.doc() = "Mesh utilities";

  py::bind_map<mathutils::mesh::MeshSamples>(m, "MeshSamples");

  m.def("tri_cycles_to_half_edge_samples",
        &mathutils::mesh::tri_cycles_to_half_edge_samples,
        "Convert triangle vertex cycles to half-edge samples",
        py::arg("V_cycle_F"),
        R"doc(
Convert triangle vertex cycles to half-edge samples.
)doc");

  //   m.def("load_vf_samples_from_ply",
  //         &mathutils::mesh::io::load_vf_samples_from_ply,
  //         "Load vertex and face samples from a PLY file",
  //         py::arg("filepath"), py::arg("preload_into_memory") = true,
  //         py::arg("verbose") = false, R"doc(
  // Load vertex and face samples from a PLY file.
  //
  // Args:
  //     filepath: Path to the PLY file
  //     preload_into_memory: Whether to preload the file into memory (default:
  //     True) verbose: Whether to print verbose output (default: False)
  //
  // Returns:
  //     A tuple of (vertices, faces) where:
  //     - vertices is an Nx3 array of vertex coordinates
  //     - faces is an Mx3 array of triangle vertex indices
  // )doc");
  //
  //   m.def("write_vf_samples_to_ply",
  //         &mathutils::mesh::io::write_vf_samples_to_ply,
  //         "Write vertex and face samples to a PLY file",
  //         py::arg("xyz_coord_V"), py::arg("V_cycle_F"), py::arg("ply_path"),
  //         py::arg("use_binary") = true, R"doc(
  //   Write vertex and face samples to a PLY file.
  //
  //   Args:
  //       xyz_coord_V: Nx3 array of vertex coordinates
  //       V_cycle_F: Mx3 array of triangle vertex indices
  //       ply_path: Output PLY file path
  //       use_binary: Whether to write binary format (default: True)
  //   )doc");
  //
  //   m.def("load_mesh_samples_from_ply",
  //         &mathutils::mesh::io::load_mesh_samples_from_ply,
  //         "Load mesh samples from a PLY file", py::arg("filepath"),
  //         py::arg("preload_into_memory") = true, py::arg("verbose") = false,
  //         R"doc(
  // Load mesh samples from a PLY file.
  // Args:
  //     filepath: Path to the PLY file
  //     preload_into_memory: Whether to preload the file into memory (default:
  //     True) verbose: Whether to print verbose output (default: False)
  // Returns:
  //     A dict of mesh samples with keys as sample names and values as
  //     ndarrays.
  // )doc");
  //
  //   m.def("write_mesh_samples_to_ply",
  //         &mathutils::mesh::io::write_mesh_samples_to_ply,
  //         "Write mesh samples to a PLY file", py::arg("mesh_samples"),
  //         py::arg("ply_path"), py::arg("use_binary") = true,
  //         R"doc(
  // Write mesh samples to a PLY file.
  // Args:
  //     mesh_samples: A dict of mesh samples with keys as sample names and
  //     values as ndarrays. ply_path: Output PLY file path use_binary: Whether
  //     to write binary format (default: True)
  // )doc");

  // m.def("save_mesh_samples", &mathutils::mesh::io::save_mesh_samples,
  //       "Write mesh samples to a PLY file", py::arg("mesh_samples"),
  //       py::arg("ply_path"), py::arg("use_binary") = true,
  //       py::arg("ply_property_convention") = "MathUtils");

  m.def(
      "save_mesh_samples",
      [](const py::dict &d, const std::string &ply_path, bool use_binary,
         const std::string &ply_property_convention) {
        mathutils::mesh::MeshSamples ms;

        for (auto item : d) {
          std::string key = py::cast<std::string>(item.first);
          mathutils::mesh::SamplesVariant val =
              py::cast<mathutils::mesh::SamplesVariant>(item.second);
          ms.insert_or_assign(std::move(key), std::move(val));
        }

        mathutils::mesh::io::save_mesh_samples(ms, ply_path, use_binary,
                                               ply_property_convention);
      },
      "Write mesh samples to a PLY file", py::arg("mesh_samples"),
      py::arg("ply_path"), py::arg("use_binary") = true,
      py::arg("ply_property_convention") = "MathUtils");

  // m.def("load_mesh_samples", &mathutils::mesh::io::load_mesh_samples,
  //       "Load mesh samples from a PLY file", py::arg("filepath"),
  //       py::arg("preload_into_memory") = true, py::arg("verbose") = false,
  //       py::arg("ply_property_convention") = "MathUtils");

  m.def(
      "load_mesh_samples",
      [](const std::string &filepath, bool preload_into_memory, bool verbose,
         const std::string &ply_property_convention) {
        auto ms = mathutils::mesh::io::load_mesh_samples(
            filepath, preload_into_memory, verbose, ply_property_convention);

        py::dict d;
        for (auto &[k, v] : ms) {
          d[py::str(k)] = py::cast(v); // SamplesVariant -> Python object
        }
        return d;
      },
      "Load mesh samples from a PLY file", py::arg("filepath"),
      py::arg("preload_into_memory") = true, py::arg("verbose") = false,
      py::arg("ply_property_convention") = "MathUtils");

  // m.def(
  //     "build_icososphere_samples",
  //     [](size_t num_refinements) {
  //       auto ms =
  //       mathutils::mesh::build_icososphere_samples(num_refinements);
  //
  //       py::dict d;
  //       for (auto &[k, v] : ms) {
  //         d[py::str(k)] = py::cast(v); // SamplesVariant -> Python object
  //       }
  //       return d;
  //     },
  //     "Refine icosohedron", py::arg("num_refinements"));

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
  using mathutils::mesh::SimplicialTopology2;

  py::class_<SimplicialTopology2>(m, "SimplicialTopology2")
      .def(py::init<>())

      .def(
          "V_cycle_e",
          [](SimplicialTopology2 &self, Index e) {
            std::span<Index> s = self.V_cycle_e(e);
            return span_view(s, py::cast(&self));
          },
          py::arg("e"),
          "Return a writable NumPy view of the vertex cycle for the edge")
      .def(
          "V_cycle_f",
          [](SimplicialTopology2 &self, Index f) {
            std::span<Index> s = self.V_cycle_f(f);
            return span_view(s, py::cast(&self));
          },
          py::arg("f"),
          "Return a writable NumPy view of the vertex cycle for the face");

  // using mathutils::mesh::SimplicialComplexBase;

  // Register base type (not meant for user-facing API)
  // py::class_<SimplicialComplexBase>(m, "_SimplicialComplexBase");

  // Public class
  py::class_<HalfEdgeTopology>(m, "HalfEdgeTopology")
      .def(py::init<>())

      .def("h_is_flippable", &HalfEdgeTopology::h_is_flippable, py::arg("h"))

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
      .def("from_mesh_samples", &HalfEdgeTopology::from_mesh_samples)

      .def("some_negative_boundary_contains_h",
           &HalfEdgeTopology::some_negative_boundary_contains_h, py::arg("h"))

      .def("some_positive_boundary_contains_h",
           &HalfEdgeTopology::some_positive_boundary_contains_h, py::arg("h"))

      .def("some_boundary_contains_h",
           &HalfEdgeTopology::some_boundary_contains_h, py::arg("h"))

      .def("some_boundary_contains_v",
           &HalfEdgeTopology::some_boundary_contains_v, py::arg("h"))

      .def("VB_cycles", &HalfEdgeTopology::VB_cycles)

      .def("flip_hedge", &HalfEdgeTopology::flip_hedge, py::arg("h"));

  py::class_<HalfEdgeMesh>(m, "HalfEdgeMesh")
      .def(py::init<>())

      .def("h_is_locally_delaunay", &HalfEdgeMesh::h_is_locally_delaunay,
           py::arg("h"))

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

      .def_property_readonly(
          "X_ambient_V",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return matrix_view(self.X_ambient_V, py::cast(&self));
          },
          "Writable NumPy view of X_ambient_V.")
      .def_property_readonly(
          "xyz_coord_V",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return matrix_view(self.X_ambient_V, py::cast(&self));
          },
          "Writable NumPy view of X_ambient_V.")

      .def_property_readonly(
          "h_out_V",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.h_out_V, py::cast(&self));
          },
          "Writable NumPy view of h_out_V.")
      .def_property_readonly(
          "h_directed_E",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.h_directed_E, py::cast(&self));
          },
          "Writable NumPy view of h_directed_E.")
      .def_property_readonly(
          "h_right_F",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.h_right_F, py::cast(&self));
          },
          "Writable NumPy view of h_right_F.")
      .def_property_readonly(
          "h_negative_B",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.h_negative_B, py::cast(&self));
          },
          "Writable NumPy view of h_negative_B.")

      .def_property_readonly(
          "v_origin_H",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.v_origin_H, py::cast(&self));
          },
          "Writable NumPy view of v_origin_H.")
      .def_property_readonly(
          "e_undirected_H",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.e_undirected_H, py::cast(&self));
          },
          "Writable NumPy view of e_undirected_H.")
      .def_property_readonly(
          "f_left_H",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.f_left_H, py::cast(&self));
          },
          "Writable NumPy view of f_left_H.")

      .def_property_readonly(
          "h_next_H",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.h_next_H, py::cast(&self));
          },
          "Writable NumPy view of h_next_H.")
      .def_property_readonly(
          "h_twin_H",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return vector_view(self.topo.h_twin_H, py::cast(&self));
          },
          "Writable NumPy view of h_twin_H.")

      .def_property_readonly(
          "V_cycle_E",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return matrix_view(self.V_cycle_E, py::cast(&self));
          },
          "Writable NumPy view of V_cycle_E.")
      .def_property_readonly(
          "V_cycle_F",
          [](mathutils::mesh::HalfEdgeMesh &self) {
            return matrix_view(self.V_cycle_F, py::cast(&self));
          },
          "Writable NumPy view of V_cycle_F.")

      .def(
          "X_ambient_v",
          [](HalfEdgeMesh &self, Index v) {
            std::span<Real> s = self.X_ambient_v(v);
            return span_view(s, py::cast(&self));
          },
          py::arg("v"),
          "Return a writable NumPy view of the vertex position row (shape "
          "(3,), etc.).")
      .def(
          "xyz_coord_v",
          [](HalfEdgeMesh &self, Index v) {
            std::span<Real> s = self.X_ambient_v(v);
            return span_view(s, py::cast(&self));
          },
          py::arg("v"),
          "Return a writable NumPy view of the vertex position row (shape "
          "(3,), etc.).")

      .def(
          "h_out_v",
          [](HalfEdgeMesh &self, Index v) { return self.topo.h_out_v(v); },
          py::arg("v"))
      .def(
          "h_directed_e",
          [](HalfEdgeMesh &self, Index e) { return self.topo.h_directed_e(e); },
          py::arg("e"))
      .def(
          "h_right_f",
          [](HalfEdgeMesh &self, Index f) { return self.topo.h_right_f(f); },
          py::arg("f"))
      .def(
          "h_negative_b",
          [](HalfEdgeMesh &self, Index b) { return self.topo.h_negative_b(b); },
          py::arg("b"))

      .def(
          "v_origin_h",
          [](HalfEdgeMesh &self, Index h) { return self.topo.v_origin_h(h); },
          py::arg("h"))
      .def(
          "e_undirected_h",
          [](HalfEdgeMesh &self, Index h) {
            return self.topo.e_undirected_h(h);
          },
          py::arg("h"))
      .def(
          "f_left_h",
          [](HalfEdgeMesh &self, Index h) { return self.topo.f_left_h(h); },
          py::arg("h"))

      .def(
          "h_next_h",
          [](HalfEdgeMesh &self, Index h) { return self.topo.h_next_h(h); },
          py::arg("h"))
      .def(
          "h_twin_h",
          [](HalfEdgeMesh &self, Index h) { return self.topo.h_twin_h(h); },
          py::arg("h"))

      .def("refresh_simplex_cycles_from_topo",
           &HalfEdgeMesh::refresh_simplex_cycles_from_topo)

      .def("split_edge", &HalfEdgeMesh::split_edge, py::arg("e"))
      .def("flip_non_delaunay", &HalfEdgeMesh::flip_non_delaunay)

      // .def("to_mesh_samples", &HalfEdgeMesh::to_mesh_samples)
      .def("to_mesh_samples",
           [](const HalfEdgeMesh &self) {
             mathutils::mesh::MeshSamples ms =
                 self.to_mesh_samples(); // copy/move return
             py::dict d;
             for (auto &[k, v] : ms) {
               d[py::str(k)] = py::cast(v); // SamplesVariant -> Python object
             }
             return d;
           })
      // .def("from_mesh_samples", &HalfEdgeMesh::from_mesh_samples)
      .def("from_mesh_samples",
           [](HalfEdgeMesh &self, const py::dict &d) {
             mathutils::mesh::MeshSamples ms;

             for (auto item : d) {
               std::string key = py::cast<std::string>(item.first);
               mathutils::mesh::SamplesVariant val =
                   py::cast<mathutils::mesh::SamplesVariant>(item.second);
               ms.insert_or_assign(std::move(key), std::move(val));
             }

             self.from_mesh_samples(ms);
           })
      .def("load_ply", &HalfEdgeMesh::load_ply,
           "Load mesh samples from a PLY file", py::arg("filepath"),
           py::arg("preload_into_memory") = true, py::arg("verbose") = false,
           py::arg("ply_property_convention") = "MathUtils")
      .def("save_ply", &HalfEdgeMesh::save_ply,
           "Save mesh samples to a PLY file", py::arg("filepath"),
           py::arg("use_binary") = true,
           py::arg("ply_property_convention") = "MathUtils");

  m.def("HalfEdgeTopology_to_SimplicialTopology2",
        &mathutils::mesh::HalfEdgeTopology_to_SimplicialTopology2,
        "Convert HalfEdgeTopology to SimplicialTopology2", py::arg("he_topo"));
  m.def("tri_cycles_to_edge_cycles",
        &mathutils::mesh::tri_cycles_to_edge_cycles,
        "Convert triangles to edges", py::arg("V_cycle_F"));
  m.def("half_edge_samples_no_edge_data_to_edge_tri_cycles",
        &mathutils::mesh::half_edge_samples_no_edge_data_to_edge_tri_cycles,
        "Convert triangles to edges", py::arg("he_samples"));

  // half_edge_samples_no_edge_data_to_edge_tri_cycles
}
