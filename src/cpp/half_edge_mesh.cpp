#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/mesh_convert_funs.hpp"

namespace mathutils {
namespace mesh {

/////////////////////////////////////////////
/////////////////////////////////////////////
// HalfEdgeTopology//////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

MeshSamples HalfEdgeTopology::to_mesh_samples() const {
  MeshSamples ms;
  // if (!h_out_V_.empty()) {
  ms["h_out_V"] = h_out_V_;
  // }
  ms["h_directed_E"] = h_directed_E_;
  ms["h_right_F"] = h_right_F_;
  ms["h_negative_B"] = h_negative_B_;
  ms["v_origin_H"] = v_origin_H_;
  ms["e_undirected_H"] = e_undirected_H_;
  ms["f_left_H"] = f_left_H_;
  ms["h_next_H"] = h_next_H_;
  ms["h_twin_H"] = h_twin_H_;
  return ms;
}

std::map<std::string_view, SamplesIndex>
HalfEdgeTopology::to_topo_samples() const {
  std::map<std::string_view, SamplesIndex> ms;
  // if (!h_out_V_.empty()) {
  ms["h_out_V"] = h_out_V_;
  // }
  ms["h_directed_E"] = h_directed_E_;
  ms["h_right_F"] = h_right_F_;
  ms["h_negative_B"] = h_negative_B_;
  ms["v_origin_H"] = v_origin_H_;
  ms["e_undirected_H"] = e_undirected_H_;
  ms["f_left_H"] = f_left_H_;
  ms["h_next_H"] = h_next_H_;
  ms["h_twin_H"] = h_twin_H_;
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

bool HalfEdgeTopology::h_is_flippable(Index h) const {
  if (some_boundary_contains_h(h)) {
    return false;
  }
  Index hlj = h;
  Index hjk = h_next_h(hlj);
  Index hli = h_next_h(h_twin_h(hlj));
  Index vi = v_head_h(hli);
  Index vk = v_head_h(hjk);
  for (auto him : generate_H_out_v_clockwise(vi)) {
    if (v_head_h(him) == vk) {
      return false;
    }
  }
  return true;
}

/////////////////////////////////////////////
/////////////////////////////////////////////
// HalfEdgeMesh//////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

bool HalfEdgeMesh::h_is_locally_delaunay(Index h) const {
  Index vi = topo.v_head_h(topo.h_next_h(topo.h_twin_h(h)));
  Index vj = topo.v_head_h(h);
  Index vk = topo.v_head_h(topo.h_next_h(h));
  Index vl = topo.v_origin_h(h);

  SamplesReal rij = X_ambient_V_.row_copy(vj) - X_ambient_V_.row_copy(vi);
  SamplesReal ril = X_ambient_V_.row_copy(vl) - X_ambient_V_.row_copy(vi);

  SamplesReal rkj = X_ambient_V_.row_copy(vj) - X_ambient_V_.row_copy(vk);
  SamplesReal rkl = X_ambient_V_.row_copy(vl) - X_ambient_V_.row_copy(vk);

  double alphai = std::acos(rij.dot(ril) / (rij.norm() * ril.norm()));
  double alphak = std::acos(rkl.dot(rkj) / (rkl.norm() * rkj.norm()));

  return alphai + alphak <= M_PI;
}

MeshSamples HalfEdgeMesh::to_mesh_samples() const {

  MeshSamples ms = topo.to_mesh_samples();

  ms["X_ambient_V"] = X_ambient_V_;
  ms["V_cycle_E"] = V_cycle_E_;
  ms["V_cycle_F"] = V_cycle_F_;
  ms.insert(attrs.begin(), attrs.end()); // doesn't remove from attrs
  return ms;
}

void HalfEdgeMesh::from_mesh_samples(MeshSamples &ms) {

  topo.from_mesh_samples(ms);

  pop_variant_to_mat_from_mesh_samples("X_ambient_V", X_ambient_V_, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_E", V_cycle_E_, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_F", V_cycle_F_, ms);

  // attrs = ms;
  for (auto &&[k, v] : ms) {
    attrs[k] = v; // overwrites / inserts
  }
}

void HalfEdgeMesh::refresh_simplex_cycles_from_topo() {

  auto s = half_edge_samples_to_edge_tri_cycles(topo.to_topo_samples());
  V_cycle_E_ = s["V_cycle_E"];
  V_cycle_F_ = s["V_cycle_F"];
}
} // namespace mesh
} // namespace mathutils
