#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/mesh_convert_funs.hpp"

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

std::map<std::string_view, SamplesIndex>
HalfEdgeTopology::to_topo_samples() const {
  std::map<std::string_view, SamplesIndex> ms;
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

  ms["X_ambient_V"] = X_ambient_V_;
  ms["V_cycle_E"] = V_cycle_E_;
  ms["V_cycle_F"] = V_cycle_F_;
  ms.insert(attrs.begin(), attrs.end());
  return ms;
}

void HalfEdgeMesh::from_mesh_samples(MeshSamples &ms) {

  topo.from_mesh_samples(ms);

  pop_variant_to_mat_from_mesh_samples("X_ambient_V", X_ambient_V_, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_E", V_cycle_E_, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_F", V_cycle_F_, ms);

  attrs = ms;
}

void HalfEdgeMesh::add_simplex_cycles_to_attrs() {

  auto s = half_edge_samples_to_edge_tri_cycles(topo.to_topo_samples());
  attrs["V_cycle_E"] = s["V_cycle_E"];
  attrs["V_cycle_F"] = s["V_cycle_F"];
}
} // namespace mesh
} // namespace mathutils
