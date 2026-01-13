// bind_mesh.cpp
#include "mathutils/bind/bind_mesh.hpp"
#include "mathutils/mesh/mesh.hpp"
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
        &mathutils::mesh_io::load_vf_samples_from_ply,
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

  m.def("write_vf_samples_to_ply", &mathutils::mesh_io::write_vf_samples_to_ply,
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
}
