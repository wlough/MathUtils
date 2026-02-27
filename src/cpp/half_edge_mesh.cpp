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
  ms["h_out_V"] = h_out_V;
  ms["h_directed_E"] = h_directed_E;
  ms["h_right_F"] = h_right_F;
  ms["h_negative_B"] = h_negative_B;
  ms["v_origin_H"] = v_origin_H;
  ms["e_undirected_H"] = e_undirected_H;
  ms["f_left_H"] = f_left_H;
  ms["h_next_H"] = h_next_H;
  ms["h_twin_H"] = h_twin_H;
  return ms;
}

std::map<std::string, SamplesIndex> HalfEdgeTopology::to_topo_samples() const {
  std::map<std::string, SamplesIndex> ms;
  ms["h_out_V"] = h_out_V;
  ms["h_directed_E"] = h_directed_E;
  ms["h_right_F"] = h_right_F;
  ms["h_negative_B"] = h_negative_B;
  ms["v_origin_H"] = v_origin_H;
  ms["e_undirected_H"] = e_undirected_H;
  ms["f_left_H"] = f_left_H;
  ms["h_next_H"] = h_next_H;
  ms["h_twin_H"] = h_twin_H;
  return ms;
}

void HalfEdgeTopology::from_mesh_samples(MeshSamples &ms) {

  pop_variant_to_mat_from_mesh_samples("h_out_V", h_out_V, ms);
  pop_variant_to_mat_from_mesh_samples("h_directed_E", h_directed_E, ms);
  pop_variant_to_mat_from_mesh_samples("h_right_F", h_right_F, ms);
  pop_variant_to_mat_from_mesh_samples("h_negative_B", h_negative_B, ms);
  pop_variant_to_mat_from_mesh_samples("v_origin_H", v_origin_H, ms);
  pop_variant_to_mat_from_mesh_samples("e_undirected_H", e_undirected_H, ms);
  pop_variant_to_mat_from_mesh_samples("f_left_H", f_left_H, ms);
  pop_variant_to_mat_from_mesh_samples("h_next_H", h_next_H, ms);
  pop_variant_to_mat_from_mesh_samples("h_twin_H", h_twin_H, ms);
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
  for (auto him : generate_H_outcw_v(vi)) {
    if (v_head_h(him) == vk) {
      return false;
    }
  }
  return true;
}

bool HalfEdgeTopology::flip_hedge(Index h) {

  if (!h_is_flippable(h)) {
    // throw std::invalid_argument("Half-edge is not flippable");
    return false;
  }
  Index h0 = h;
  Index h1 = h_twin_h(h0);
  Index h2 = h_next_h(h0);
  Index h3 = h_next_h(h2);
  Index h4 = h_next_h(h1);
  Index h5 = h_next_h(h4);
  Index v0 = v_origin_h(h1);
  Index v1 = v_origin_h(h3);
  Index v2 = v_origin_h(h0);
  Index v3 = v_origin_h(h5);
  Index f0 = f_left_h(h0);
  Index f1 = f_left_h(h1);
  //   update vertices
  if (h_out_v(v0) == h1) {
    h_out_V[v0] = h2;
  }
  if (h_out_v(v2) == h0) {
    h_out_V[v2] = h4;
  }
  // update half-edges
  // v_origin_H
  // h_next_H
  // h_twin_H
  // f_left_H
  // update_mat_h(h0, v3, h3, std::nullopt, std::nullopt);
  v_origin_H[h0] = v3;
  h_next_H[h0] = h3;
  // update_mat_h(h1, v1, h5, std::nullopt, std::nullopt);
  v_origin_H[h1] = v1;
  h_next_H[h1] = h5;
  // update_mat_h(h2, std::nullopt, h1, std::nullopt, f1);
  h_next_H[h2] = h1;
  f_left_H[h2] = f1;
  // update_mat_h(h3, std::nullopt, h4, std::nullopt, std::nullopt);
  h_next_H[h3] = h4;
  // update_mat_h(h4, std::nullopt, h0, std::nullopt, f0);
  h_next_H[h4] = h0;
  f_left_H[h4] = f0;
  // update_mat_h(h5, std::nullopt, h2, std::nullopt, std::nullopt);
  h_next_H[h5] = h2;
  // update faces
  if (h_right_f(f0) == h2) {
    h_right_F[f0] = h3;
  }
  if (h_right_f(f1) == h4) {
    h_right_F[f1] = h5;
  }
  return true;
}

std::vector<SamplesIndex> HalfEdgeTopology::VB_cycles() const {
  size_t Nb = num_boundaries();
  std::vector<SamplesIndex> all_V_cycle_B;
  all_V_cycle_B.reserve(Nb);
  for (Index b = 0; b < Nb; b++) {
    std::vector<Index> Vnegative;
    for (auto h : generate_H_next_h(h_negative_b(b))) {
      Vnegative.push_back(v_origin_h(h));
    }
    Index Nv = Vnegative.size();
    all_V_cycle_B.emplace_back(SamplesIndex(Nv));
    for (Index _i = 0; _i < Nv; _i++) {
      Index i = Nv - 1 - _i;
      all_V_cycle_B.back()[i] = Vnegative[_i];
    }
  }
  return all_V_cycle_B;
}

/////////////////////////////////////////////
/////////////////////////////////////////////
// HalfEdgeMesh /////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

bool HalfEdgeMesh::h_is_locally_delaunay(Index h) const {
  Index vi = topo.v_head_h(topo.h_next_h(topo.h_twin_h(h)));
  Index vj = topo.v_head_h(h);
  Index vk = topo.v_head_h(topo.h_next_h(h));
  Index vl = topo.v_origin_h(h);

  SamplesReal rij = X_ambient_V.row_copy(vj) - X_ambient_V.row_copy(vi);
  SamplesReal ril = X_ambient_V.row_copy(vl) - X_ambient_V.row_copy(vi);

  SamplesReal rkj = X_ambient_V.row_copy(vj) - X_ambient_V.row_copy(vk);
  SamplesReal rkl = X_ambient_V.row_copy(vl) - X_ambient_V.row_copy(vk);

  double alphai = std::acos(rij.dot(ril) / (rij.norm() * ril.norm()));
  double alphak = std::acos(rkl.dot(rkj) / (rkl.norm() * rkj.norm()));

  return alphai + alphak <= M_PI;
}

MeshSamples HalfEdgeMesh::to_mesh_samples() const {

  MeshSamples ms = topo.to_mesh_samples();

  ms["X_ambient_V"] = X_ambient_V;
  ms["V_cycle_E"] = V_cycle_E;
  ms["V_cycle_F"] = V_cycle_F;
  ms.insert(attrs.begin(), attrs.end()); // doesn't remove from attrs
  return ms;
}

void HalfEdgeMesh::from_mesh_samples(MeshSamples &ms) {

  topo.from_mesh_samples(ms);

  pop_variant_to_mat_from_mesh_samples("X_ambient_V", X_ambient_V, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_E", V_cycle_E, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_F", V_cycle_F, ms);

  // attrs = ms;
  for (auto &&[k, v] : ms) {
    attrs[k] = v; // overwrites / inserts
  }
}

void HalfEdgeMesh::refresh_simplex_cycles_from_topo() {

  auto s = half_edge_samples_to_edge_tri_cycles(topo.to_topo_samples());
  V_cycle_E = s["V_cycle_E"];
  V_cycle_F = s["V_cycle_F"];
}

/**
 * @brief Divides face by adding a new vertex at midpoint of an edge
 *
 * @param f
 * @details
 * ```
 *                 v2                                    v2
 *               /   \                                 / | \
 *              /     \                               /  |  \
 *             /       \                             /   |   \
 *            /         \                           /    |    \
 *           /           \                         /     |     \
 *          /             \                       /      |      \
 *         /               \                     /       |       \
 *        /e2             e1\                   /h2      |      h1\
 *       /        f0         \                 /         |e2       \
 *      /                     \               /        h4|h5        \
 *     /                       \             /    f0     |    f1     \
 *    /                         \           /            |            \
 *   /            e0             \         /    e0       |      e1     \
 *  /             h0              \ ----> /     h0       |      h3      \
 * v0 ----------------------------v1     v0 ------------v3--------------v1
 * ```
 */
void HalfEdgeMesh::split_edge(Index e) {

  Index h0 = topo.h_directed_e(e);
  Index ht0 = topo.h_twin_h(h0);

  if (topo.some_negative_boundary_contains_h(h0)) {
    std::swap(h0, ht0);
  }

  Index h1 = topo.h_next_h(h0);
  Index h2 = topo.h_next_h(h1);
  Index v0 = topo.v_origin_h(h0);
  Index v1 = topo.v_origin_h(h1);
  Index v2 = topo.v_origin_h(h2);
  Index e0 = topo.e_undirected_h(h0);
  Index f0 = topo.f_left_h(h0);

  // new
  Index h3 = topo.num_half_edges();
  Index h4 = h3 + 1;
  Index h5 = h4 + 1;
  Index v3 = topo.num_vertices();
  Index e1 = topo.num_edges();
  Index e2 = e1 + 1;
  Index f1 = topo.num_faces();

  SamplesReal X_ambient_v3 =
      (X_ambient_V.row_copy(v0) + X_ambient_V.row_copy(v1)) / 2.0;

  // h0:
  // Index h_twin_h0 = ;
  Index h_next_h0 = h4;
  // Index v_origin_h0 = same;
  // Index e_undirected_h0 = same;
  // Index f_left_h0 = same;

  // h1
  // Index h_twin_h1 = same;
  Index h_next_h1 = h5;
  // Index v_origin_h1 = same;
  // Index e_undirected_h1 = same;
  Index f_left_h1 = f1;

  // h2
  // Index h_twin_h2 = same;
  // Index h_next_h2 = same;
  // Index v_origin_h2 = same;
  // Index e_undirected_h2 = same;
  // Index f_left_h2 = same;

  // h3
  // Index h_twin_h3 = ;
  Index h_next_h3 = h1;
  Index v_origin_h3 = v3;
  // Index e_undirected_h3 = ;
  Index f_left_h3 = f1;

  // h4
  Index h_twin_h4 = h5;
  Index h_next_h4 = h2;
  Index v_origin_h4 = v3;
  Index e_undirected_h4 = e2;
  Index f_left_h4 = f0;

  // h5
  Index h_twin_h5 = h4;
  Index h_next_h5 = h3;
  Index v_origin_h5 = v2;
  Index e_undirected_h5 = e2;
  Index f_left_h5 = f1;

  if (topo.some_negative_boundary_contains_h(ht0)) {
    //
    // do stuff
    return;
  }

  //
  // do other face
  Index ht1 = topo.h_next_h(ht0);
  Index ht2 = topo.h_next_h(ht1);
  Index vt0 = topo.v_origin_h(ht0);
  Index vt1 = topo.v_origin_h(ht1);
  Index vt2 = topo.v_origin_h(ht2);
  Index et0 = topo.e_undirected_h(ht0);
  Index ft0 = topo.f_left_h(ht0);

  // new
  // Nh = h5+1
  // Nv = v3+1
  // Ne = e2+1
  // Nf = f1+1
  Index ht3 = h5 + 1;
  Index ht4 = ht3 + 1;
  Index ht5 = ht4 + 1;
  Index vt3 = v3 + 1;
  Index et1 = e2 + 1;
  Index et2 = et1 + 1;
  Index ft1 = f1 + 1;

  // h0:
  Index h_twin_h0 = ht3;

  // h3
  Index h_twin_h3 = ht0;
  // Index e_undirected_h3 = ;
}

} // namespace mesh
} // namespace mathutils
