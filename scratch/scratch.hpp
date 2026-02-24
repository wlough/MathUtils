.def_property_readonly(
    "h_twin_H",
    [](mathutils::mesh::HalfEdgeMesh &self) {
      return matrix_view(self.topo.h_twin_H_, py::cast(&self));
    },
    "Writable NumPy view of h_twin_H.")
