#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include <unordered_set>

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

Index HalfEdgeTopology::h_prev_h(Index h) const {
  Index h_next = h_next_H[h];
  int guard = 0;
  while (h_next != h) {
    h = h_next;
    h_next = h_next_H[h];
    guard++;
    assert(guard < 100);
  }
  return h;
}

Index HalfEdgeTopology::h_prev_h_by_rot(Index h) const {
  Index h_rot = h_rotcw_h(h);
  int guard = 0;
  while (h_rot != h) {
    h = h_rot;
    h_rot = h_rotcw_h(h);
    guard++;
    assert(guard < 100);
  }
  return h_twin_H[h];
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

bool HalfEdgeTopology::h_is_collapsable(Index h) const {

  Index h0 = h;
  Index h1 = h_next_H[h0];
  Index h2 = h_next_H[h1];
  Index h3 = h_twin_H[h0];
  Index h4 = h_next_H[h3];
  Index h5 = h_next_H[h4];

  Index v0 = v_origin_H[h0];
  Index v1 = v_origin_H[h1];
  Index v2 = v_origin_H[h2];
  Index v3 = v_origin_H[h5];

  if (some_boundary_contains_v(v0) || some_boundary_contains_v(v1)) {
    return false;
  }

  std::unordered_set<Index> V1;
  for (auto v : generate_V_adjacent_v(v1)) {
    V1.insert(v);
  }
  std::unordered_set<Index> V0_intersect_V1;
  for (auto v : generate_V_adjacent_v(v0)) {
    if (V1.find(v) == V1.end()) {
      continue;
    }
    V0_intersect_V1.insert(v);
  }
  if (V0_intersect_V1.size() != 2 ||
      V0_intersect_V1.find(v2) == V0_intersect_V1.end() ||
      V0_intersect_V1.find(v3) == V0_intersect_V1.end()) {
    return false;
  }

  return true;
}

bool HalfEdgeTopology::collapse_hedge(Index h) {
  if (!h_is_collapsable(h)) {
    return false;
  }

  Index h0 = h;
  Index h1 = h_next_H[h0];
  Index h2 = h_next_H[h1];
  Index h3 = h_twin_H[h0];
  Index h4 = h_next_H[h3];
  Index h5 = h_next_H[h4];

  Index v0 = v_origin_H[h0];
  Index v1 = v_origin_H[h1];
  Index v2 = v_origin_H[h2];
  Index v3 = v_origin_H[h5];

  Index e0 = e_undirected_H[h0];
  Index e1 = e_undirected_H[h1];
  Index e2 = e_undirected_H[h2];
  Index e3 = e_undirected_H[h4];
  Index e4 = e_undirected_H[h5];

  Index f0 = f_left_H[h0];
  Index f1 = f_left_H[h3];

  // swap_v_indices(v0, num_vertices() - 1); // need to update positions...

  swap_e_indices(e0, num_vertices() - 1);
  swap_e_indices(e1, num_vertices() - 2);
  swap_e_indices(e2, num_vertices() - 3);

  swap_f_indices(f0, num_faces() - 1);
  swap_f_indices(f1, num_faces() - 2);

  // swap_h_indices(h0, num_half_edges() - 1);
  // swap_h_indices(h1, num_half_edges() - 2);
  // swap_h_indices(h2, num_half_edges() - 3);
  // swap_h_indices(h3, num_half_edges() - 4);
  // swap_h_indices(h4, num_half_edges() - 5);
  // swap_h_indices(h5, num_half_edges() - 6);

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

void HalfEdgeMesh::split_edge(Index e) {

  Index h0 = topo.h_directed_e(e);
  Index ht0 = topo.h_twin_h(h0);

  if (topo.some_negative_boundary_contains_h(h0)) {
    std::swap(h0, ht0);
  }

  // If e is in a boundary, only split one face
  if (topo.some_negative_boundary_contains_h(ht0)) {
    // existing
    Index h1 = topo.h_next_h(h0);
    Index h2 = topo.h_next_h(h1);
    Index ht1 = topo.h_next_h(ht0);
    Index ht2 = topo.h_prev_h_by_rot(ht0);

    Index v0 = topo.v_origin_h(h0);
    Index v1 = topo.v_origin_h(h1);
    Index v2 = topo.v_origin_h(h2);
    Index e0 = topo.e_undirected_h(h0);
    Index f0 = topo.f_left_h(h0);
    Index ft0 = topo.f_left_h(ht0); // -b0-1

    // new
    Index h3 = topo.num_half_edges();
    Index h4 = h3 + 1;
    Index h5 = h4 + 1;
    Index ht3 = h5 + 1;

    Index v3 = topo.num_vertices();
    Index e1 = topo.num_edges();
    Index e2 = e1 + 1;
    Index f1 = topo.num_faces();

    SamplesReal x3 =
        (X_ambient_V.row_copy(v0) + X_ambient_V.row_copy(v1)) / 2.0;

    size_t Nh = topo.num_half_edges() + 4;
    size_t Nv = topo.num_vertices() + 1;
    size_t Ne = topo.num_edges() + 2;
    size_t Nf = topo.num_faces() + 1;

    topo.h_next_H.conservativeResize(Nh);
    topo.h_twin_H.conservativeResize(Nh);
    topo.v_origin_H.conservativeResize(Nh);
    topo.e_undirected_H.conservativeResize(Nh);
    topo.f_left_H.conservativeResize(Nh);

    X_ambient_V.conservativeResize(Nv, 3);
    topo.h_out_V.conservativeResize(Nv);

    topo.h_directed_E.conservativeResize(Ne);

    topo.h_right_F.conservativeResize(Nf);

    /////////////////////////////
    // Update existing h0 side //
    /////////////////////////////
    // h0:
    topo.h_twin_H[h0] = ht3;
    topo.h_next_H[h0] = h4;
    // topo.v_origin_H[h0] = same;
    // topo.e_undirected_H[h0] = same;
    // topo.f_left_H[h0] = same;
    // h1:
    // topo.h_twin_H[h1] = same;
    topo.h_next_H[h1] = h5;
    // topo.v_origin_H[h1] = same;
    // topo.e_undirected_H[h1] = same;
    topo.f_left_H[h1] = f1;
    // h2:
    // topo.h_twin_H[h2] = same;
    // topo.h_next_H[h2] = same;
    // topo.v_origin_H[h2] = same;
    // topo.e_undirected_H[h2] = same;
    // topo.f_left_H[h2] = same;
    // v0: same
    // v1: same
    // v2: same
    // e0: same
    // f0:
    if (topo.h_right_F[f0] == h1) {
      topo.h_right_F[f0] = h0;
    }
    //////////////////////////////
    // Update existing ht0 side //
    //////////////////////////////
    // ht0:
    topo.h_twin_H[ht0] = h3;
    topo.h_next_H[ht0] = ht3;
    // topo.v_origin_H[ht0] = same;
    topo.e_undirected_H[ht0] = e1;
    // topo.f_left_H[ht0] = same;
    // ht1:
    // topo.h_twin_H[ht1] = same;
    // topo.h_next_H[ht1] = same;
    // topo.v_origin_H[ht1] = same;
    // topo.e_undirected_H[ht1] = same;
    // topo.f_left_H[ht1] = same;
    // ht2:
    // topo.h_twin_H[ht2] = same;
    // topo.h_next_H[ht2] = same;
    // topo.v_origin_H[ht2] = same;
    // topo.e_undirected_H[ht2] = same;
    // topo.f_left_H[ht2] = same;
    ////////////////////////
    // Update new h0 side //
    ////////////////////////
    // h3:
    topo.h_twin_H[h3] = ht0;
    topo.h_next_H[h3] = h1;
    topo.v_origin_H[h3] = v3;
    topo.e_undirected_H[h3] = e1;
    topo.f_left_H[h3] = f1;
    // h4:
    topo.h_twin_H[h4] = h5;
    topo.h_next_H[h4] = h2;
    topo.v_origin_H[h4] = v3;
    topo.e_undirected_H[h4] = e2;
    topo.f_left_H[h4] = f0;
    // h5:
    topo.h_twin_H[h5] = h4;
    topo.h_next_H[h5] = h3;
    topo.v_origin_H[h5] = v2;
    topo.e_undirected_H[h5] = e2;
    topo.f_left_H[h5] = f1;
    // v3:
    topo.h_out_V[v3] = h3;
    X_ambient_V.set_row(v3, {x3[0], x3[1], x3[2]});
    // e1:
    topo.h_directed_E[e1] = h3;
    // e2:
    topo.h_directed_E[e2] = h4;
    // f1:
    topo.h_right_F[f1] = h3;
    /////////////////////////
    // Update new ht0 side //
    /////////////////////////
    // ht3:
    topo.h_twin_H[ht3] = h0;
    topo.h_next_H[ht3] = ht1;
    topo.v_origin_H[ht3] = v3;
    topo.e_undirected_H[ht3] = e0;
    topo.f_left_H[ht3] = ft0;
    return;
  }

  // existing
  Index h1 = topo.h_next_h(h0);
  Index h2 = topo.h_next_h(h1);

  Index ht1 = topo.h_next_h(ht0);
  // Index ht2 = topo.h_prev_h(ht0);
  Index ht2 = topo.h_next_h(topo.h_next_h(ht0));

  Index v0 = topo.v_origin_h(h0);
  Index v1 = topo.v_origin_h(h1);
  Index v2 = topo.v_origin_h(h2);

  Index vt2 = topo.v_origin_h(ht2);

  Index e0 = topo.e_undirected_h(h0);

  Index f0 = topo.f_left_h(h0);

  Index ft0 = topo.f_left_h(ht0);

  // new
  Index h3 = topo.num_half_edges();
  Index h4 = h3 + 1;
  Index h5 = h4 + 1;

  Index ht3 = h5 + 1;
  Index ht4 = ht3 + 1;
  Index ht5 = ht4 + 1;

  Index v3 = topo.num_vertices();

  Index e1 = topo.num_edges();
  Index e2 = e1 + 1;

  Index et2 = e2 + 1;

  Index f1 = topo.num_faces();

  Index ft1 = f1 + 1;

  SamplesReal x3 = (X_ambient_V.row_copy(v0) + X_ambient_V.row_copy(v1)) / 2.0;

  size_t Nh = topo.num_half_edges() + 6;
  size_t Nv = topo.num_vertices() + 1;
  size_t Ne = topo.num_edges() + 3;
  size_t Nf = topo.num_faces() + 2;

  topo.h_next_H.conservativeResize(Nh);
  topo.h_twin_H.conservativeResize(Nh);
  topo.v_origin_H.conservativeResize(Nh);
  topo.e_undirected_H.conservativeResize(Nh);
  topo.f_left_H.conservativeResize(Nh);

  X_ambient_V.conservativeResize(Nv, 3);
  topo.h_out_V.conservativeResize(Nv);

  topo.h_directed_E.conservativeResize(Ne);

  topo.h_right_F.conservativeResize(Nf);

  /////////////////////////////
  // Update existing h0 side //
  /////////////////////////////
  // h0:
  topo.h_twin_H[h0] = ht3;
  topo.h_next_H[h0] = h4;
  // topo.v_origin_H[h0] = same;
  // topo.e_undirected_H[h0] = same;
  // topo.f_left_H[h0] = same;
  // h1:
  // topo.h_twin_H[h1] = same;
  topo.h_next_H[h1] = h5;
  // topo.v_origin_H[h1] = same;
  // topo.e_undirected_H[h1] = same;
  topo.f_left_H[h1] = f1;
  // h2:
  // topo.h_twin_H[h2] = same;
  // topo.h_next_H[h2] = same;
  // topo.v_origin_H[h2] = same;
  // topo.e_undirected_H[h2] = same;
  // topo.f_left_H[h2] = same;
  // v0: same
  // v1: same
  // v2: same
  // e0: same
  // f0:
  if (topo.h_right_F[f0] == h1) {
    topo.h_right_F[f0] = h0;
  }
  //////////////////////////////
  // Update existing ht0 side //
  //////////////////////////////
  // ht0:
  topo.h_twin_H[ht0] = h3;
  topo.h_next_H[ht0] = ht4;
  // topo.v_origin_H[ht0] = same;
  topo.e_undirected_H[ht0] = e1;
  // topo.f_left_H[ht0] = same;
  // ht1:
  // topo.h_twin_H[ht1] = same;
  topo.h_next_H[ht1] = ht5;
  // topo.v_origin_H[ht1] = same;
  // topo.e_undirected_H[ht1] = same;
  topo.f_left_H[ht1] = ft1;
  // ht2:
  // topo.h_twin_H[ht2] = same;
  // topo.h_next_H[ht2] = same;
  // topo.v_origin_H[ht2] = same;
  // topo.e_undirected_H[ht2] = same;
  // topo.f_left_H[ht2] = same;
  // vt2: same
  // ft0:
  if (topo.h_right_F[ft0] == ht1) {
    topo.h_right_F[ft0] = ht0;
  }
  ////////////////////////
  // Update new h0 side //
  ////////////////////////
  // h3:
  topo.h_twin_H[h3] = ht0;
  topo.h_next_H[h3] = h1;
  topo.v_origin_H[h3] = v3;
  topo.e_undirected_H[h3] = e1;
  topo.f_left_H[h3] = f1;
  // h4:
  topo.h_twin_H[h4] = h5;
  topo.h_next_H[h4] = h2;
  topo.v_origin_H[h4] = v3;
  topo.e_undirected_H[h4] = e2;
  topo.f_left_H[h4] = f0;
  // h5:
  topo.h_twin_H[h5] = h4;
  topo.h_next_H[h5] = h3;
  topo.v_origin_H[h5] = v2;
  topo.e_undirected_H[h5] = e2;
  topo.f_left_H[h5] = f1;
  // v3:
  topo.h_out_V[v3] = h3;
  X_ambient_V.set_row(v3, {x3[0], x3[1], x3[2]});
  // e1:
  topo.h_directed_E[e1] = h3;
  // e2:
  topo.h_directed_E[e2] = h4;
  // f1:
  topo.h_right_F[f1] = h3;
  /////////////////////////
  // Update new ht0 side //
  /////////////////////////
  // ht3:
  topo.h_twin_H[ht3] = h0;
  topo.h_next_H[ht3] = ht1;
  topo.v_origin_H[ht3] = v3;
  topo.e_undirected_H[ht3] = e0;
  topo.f_left_H[ht3] = ft1;
  // ht4:
  topo.h_twin_H[ht4] = ht5;
  topo.h_next_H[ht4] = ht2;
  topo.v_origin_H[ht4] = v3;
  topo.e_undirected_H[ht4] = et2;
  topo.f_left_H[ht4] = ft0;
  // ht5:
  topo.h_twin_H[ht5] = ht4;
  topo.h_next_H[ht5] = ht3;
  topo.v_origin_H[ht5] = vt2;
  topo.e_undirected_H[ht5] = et2;
  topo.f_left_H[ht5] = ft1;
  // et2:
  topo.h_directed_E[et2] = ht4;
  // ft1:
  topo.h_right_F[ft1] = ht3;
}

int HalfEdgeMesh::flip_non_delaunay() {
  int count = 0;
  std::vector<size_t> E = rng_.random_permutation(topo.num_edges());
  for (auto e : E) {
    Index h = topo.h_directed_e(e);
    if (!h_is_locally_delaunay(h)) {
      topo.flip_hedge(h);
      ++count;
    }
  }
  return count;
}

// bool HalfEdgeMesh::collapse_edge(Index e) {
//   Index h = topo.h_directed_e(e);
//
//   return topo.collapse_hedge(h);
// }
} // namespace mesh
} // namespace mathutils
