#include "mathutils/mesh/half_edge_mesh.hpp"

namespace mathutils {
namespace mesh {

MeshSamples HalfEdgeTopology::to_mesh_samples() const {
  MeshSamples ms;
  if (!h_out_V_.empty()) {
    ms["h_out_V"] = h_out_V_;
  }
  if (!h_directed_E_.empty()) {
    ms["h_directed_E"] = h_directed_E_;
  }
  if (!h_right_F_.empty()) {
    ms["h_right_F"] = h_right_F_;
  }
  if (!h_negative_B_.empty()) {
    ms["h_negative_B"] = h_negative_B_;
  }
  if (!v_origin_H_.empty()) {
    ms["v_origin_H"] = v_origin_H_;
  }
  if (!e_undirected_H_.empty()) {
    ms["e_undirected_H"] = e_undirected_H_;
  }
  if (!f_left_H_.empty()) {
    ms["f_left_H"] = f_left_H_;
  }
  if (!h_next_H_.empty()) {
    ms["h_next_H"] = h_next_H_;
  }
  if (!h_twin_H_.empty()) {
    ms["h_twin_H"] = h_twin_H_;
  }

  return ms;
}

void HalfEdgeTopology::from_mesh_samples(MeshSamples &ms) {

  pop_variant_to_mat_from_mesh_samples("h_out_V", h_out_V_, ms);
  pop_variant_to_mat_from_mesh_samples("h_directed_E", h_directed_E_, ms);
  pop_variant_to_mat_from_mesh_samples("h_right_F", h_right_F_, ms);
  pop_variant_to_mat_from_mesh_samples("h_negative_B", h_negative_B_, ms);
  pop_variant_to_mat_from_mesh_samples("v_origin_H", v_origin_H_, ms);
  pop_variant_to_mat_from_mesh_samples("e_undirected_H", e_undirected_H_, ms);
  pop_variant_to_mat_from_mesh_samples("f_left_H", f_left_H_, ms);
  pop_variant_to_mat_from_mesh_samples("h_next_H", h_next_H_, ms);
  pop_variant_to_mat_from_mesh_samples("h_twin_H", h_twin_H_, ms);
}

MeshSamples HalfEdgeMesh::to_mesh_samples() const {

  MeshSamples ms = topo.to_mesh_samples();

  if (!X_ambient_V_.empty()) {
    ms["X_ambient_V"] = X_ambient_V_;
  }
  ms.insert(attrs.begin(), attrs.end());
  return ms;
}

void HalfEdgeMesh::from_mesh_samples(MeshSamples &ms) {

  topo.from_mesh_samples(ms);

  pop_variant_to_mat_from_mesh_samples("X_ambient_V", X_ambient_V_, ms);

  attrs = ms;
}

} // namespace mesh
} // namespace mathutils
