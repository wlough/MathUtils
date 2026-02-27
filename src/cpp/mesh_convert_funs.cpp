/**
 * @file mesh.cpp
 */
#include "mathutils/mesh/mesh_convert_funs.hpp"
#include "mathutils/hash.hpp" // mathutils::hash::ArrayHash, mathutils::hash::hash_combine
#include "mathutils/mesh/mesh_common.hpp"
#include <map>
#include <unordered_set>
#include <vector>
// #include <array>
// #include <cstddef>
// #include <tuple>
// #include <unordered_map>

namespace mathutils {
namespace mesh {

Index find_halfedge_index_of_twin(const SamplesIndex &H, const Index &h) {
  Index v0 = H(h, 0);
  Index v1 = H(h, 1);
  for (Index h_twin = 0; h_twin < H.rows(); ++h_twin) {
    if ((H(h_twin, 0) == v1) && (H(h_twin, 1) == v0)) {
      return h_twin;
    }
  }
  return InvalidIndex;
}

SamplesIndex tri_cycles_to_edge_cycles(const SamplesIndex &V_cycle_F) {
  Index Nf = V_cycle_F.rows();
  using Edge = std::array<Index, 2>;
  using EdgeHash = mathutils::hash::ArrayHash<Index, 2>;

  auto sort_edge = [](Index a, Index b) -> Edge {
    if (b < a)
      std::swap(a, b);
    return Edge{a, b};
  };

  std::unordered_set<Edge, EdgeHash> E;
  E.reserve(3 * Nf);
  SamplesIndex V_cycle_E;

  for (Index f{0}; f < Nf; ++f) {
    Index i = V_cycle_F(f, 0);
    Index j = V_cycle_F(f, 1);
    Index k = V_cycle_F(f, 2);

    E.insert(sort_edge(i, j));
    E.insert(sort_edge(j, k));
    E.insert(sort_edge(i, k));
  }

  // with sorting O(F + E*log(E))
  Index Ne = E.size();
  std::vector<Edge> E_sorted;
  E_sorted.reserve(Ne);
  for (const auto &e : E)
    E_sorted.push_back(e);

  std::sort(E_sorted.begin(), E_sorted.end(), [](const Edge &a, const Edge &b) {
    return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
  });

  V_cycle_E.resize(Ne, 2);
  for (Index e{0}; e < Ne; ++e) {
    V_cycle_E.set_row(e, {E_sorted[e][0], E_sorted[e][1]});
  }
  return V_cycle_E;

  // without sorting O(F)
  // size_t Ne = E.size();
  // V_cycle_E.resize(Ne, 2);
  // Index e{0};
  // for (const Edge &edge : E) {
  //   V_cycle_E.set_row(e++, {edge[0], edge[1]});
  // }
  // return V_cycle_E;
}

std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples_alt(const SamplesIndex &V_cycle_F) {
  size_t Nf = V_cycle_F.rows();
  using Edge = std::array<Index, 2>;
  using EdgeHash = mathutils::hash::ArrayHash<Index, 2>;

  auto sort_edge = [](Index a, Index b) -> Edge {
    if (b < a)
      std::swap(a, b);
    return Edge{a, b};
  };

  std::unordered_set<Edge, EdgeHash> Edges;
  Edges.reserve(3 * Nf);
  std::unordered_map<Edge, std::set<Index>, EdgeHash> Fset_of_Edge;
  Fset_of_Edge.reserve(3 * Nf);
  std::vector<std::unordered_set<Edge, EdgeHash>> EdgeSet_of_F(Nf);

  for (Index f{0}; f < Nf; ++f) {
    Index i = V_cycle_F(f, 0);
    Index j = V_cycle_F(f, 1);
    Index k = V_cycle_F(f, 2);

    Edge ij = sort_edge(i, j);
    Edge jk = sort_edge(j, k);
    Edge ik = sort_edge(i, k);

    Edges.insert(ij);
    Edges.insert(jk);
    Edges.insert(ik);

    Fset_of_Edge[ij].insert(f);
    Fset_of_Edge[jk].insert(f);
    Fset_of_Edge[ik].insert(f);

    EdgeSet_of_F[f].insert(ij);
    EdgeSet_of_F[f].insert(jk);
    EdgeSet_of_F[f].insert(ik);
  }

  // with sorting O(F + E*log(E))
  size_t Ne = Edges.size();
  std::vector<Edge> Edges_sorted;
  Edges_sorted.reserve(Ne);
  for (const auto &e : Edges) {
    Edges_sorted.push_back(e);
  }

  std::sort(Edges_sorted.begin(), Edges_sorted.end(),
            [](const Edge &a, const Edge &b) {
              return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
            });

  std::vector<std::set<Index>> F_of_E_sorted;
  F_of_E_sorted.reserve(Ne);
  for (const auto &e : Edges_sorted) {
    F_of_E_sorted.push_back(Fset_of_Edge[e]);
  }

  std::unordered_map<Edge, Index, EdgeHash> E_of_Edge;
  E_of_Edge.reserve(Ne);
  for (Index e{0}; e < Ne; ++e) {
    E_of_Edge[Edges_sorted[e]] = e;
  }

  std::vector<std::set<Index>> E_of_F_sorted(Nf);
  for (Index f{0}; f < Nf; ++f) {
    // Ed
  }

  SamplesIndex V_cycle_E(Ne, 2);
  for (Index e{0}; e < Ne; ++e) {
    V_cycle_E.set_row(e, {Edges_sorted[e][0], Edges_sorted[e][1]});
  }

  size_t Nv = V_cycle_F.maxCoeff() + 1;
  size_t Nh = 2 * Ne;

  SamplesIndex h_out_V(Nv);
  SamplesIndex v_origin_H(Nh);
  SamplesIndex h_next_H(Nh);
  SamplesIndex h_twin_H(Nh);
  SamplesIndex f_left_H(Nh);
  f_left_H.fill(InvalidIndex);
  SamplesIndex h_right_F(Nf);
  SamplesIndex e_undirected_H(Nh);
  SamplesIndex h_directed_E(Ne);
  SamplesIndex h_negative_B;

  for (Index e{0}; e < Ne; ++e) {
    Index v0 = V_cycle_E(e, 0);
    Index v1 = V_cycle_E(e, 1);
    Index h = 2 * e;
    Index ht = 2 * e + 1;

    h_directed_E[e] = h;

    h_twin_H[h] = ht;
    v_origin_H[h] = v0;
    e_undirected_H[h] = e;
    h_out_V[v0] = h;

    h_twin_H[ht] = h;
    v_origin_H[ht] = v1;
    e_undirected_H[ht] = e;
    h_out_V[v1] = ht;

    Index f0;
    Index f1 = InvalidIndex;
    if (F_of_E_sorted[e].size() == 2) {
      auto it = F_of_E_sorted[e].begin();
      f0 = *it++;
      f1 = *it;
    } else if (F_of_E_sorted[e].size() != 1) {
      throw std::runtime_error("Expected 1 or 2 elements");
    }

    Index i = V_cycle_F(f0, 0);
    Index j = V_cycle_F(f0, 1);
    Index k = V_cycle_F(f0, 2);

    if ((v0 == i && v1 == j) || (v0 == j && v1 == k) || (v0 == k && v1 == i)) {
      f_left_H[h] = f0;
      f_left_H[ht] = f1;
    } else if ((v1 == i && v0 == j) || (v1 == j && v0 == k) ||
               (v1 == k && v0 == i)) {
      f_left_H[h] = f1;
      f_left_H[ht] = f0;
    } else {
      throw std::runtime_error("Half edge not in face");
    }
  }

  auto find_face_index = [&](Index h) -> Index {
    Index f_left = InvalidIndex;
    for (Index f{0}; f < Nf; ++f) {
      std::set<Index> face({V_cycle_F(f, 0), V_cycle_F(f, 1), V_cycle_F(f, 2)});
    }
    return f_left;
  };

  for (Index f{0}; f < Nf; ++f) {
    Index v0 = V_cycle_F(f, 0);
    Index v1 = V_cycle_F(f, 1);
    Index v2 = V_cycle_F(f, 2);

    // find h0 with

    // Index v_origin = v0;
    // Index v_head = v1;
    //
    // Index h = h_out_V[v_origin];
    // Index v = h_out_V[h_twin_H[h]];
    // while (v != v_head) {
    //   //
    // }
  }

  std::map<std::string, SamplesIndex> data_map;

  data_map["h_out_V"] = h_out_V;
  data_map["h_directed_E"] = h_directed_E;
  data_map["h_right_F"] = h_right_F;
  data_map["h_negative_B"] = h_negative_B;

  data_map["v_origin_H"] = v_origin_H;
  data_map["e_undirected_H"] = e_undirected_H;
  data_map["f_left_H"] = f_left_H;

  data_map["h_next_H"] = h_next_H;
  data_map["h_twin_H"] = h_twin_H;

  return data_map;
}

std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples(const SamplesIndex &V_cycle_F) {

  Index Nv = V_cycle_F.maxCoeff() + 1;
  Index Nf = V_cycle_F.rows();
  // num interior + num positive boundary half-edges
  Index Nh0 = 3 * Nf;
  SamplesIndex H0(Nh0, 2);
  // half-edge samples
  // h_out=Nh0 if not assigned
  // h_twin=InvalidIndex if not assigned
  SamplesIndex h_out_V(Nv);
  h_out_V.fill(Nh0);
  SamplesIndex v_origin_H(Nh0);
  SamplesIndex h_next_H(Nh0);
  SamplesIndex h_twin_H(Nh0);
  h_twin_H.fill(InvalidIndex);
  SamplesIndex f_left_H(Nh0);
  SamplesIndex h_right_F(Nf);
  SamplesIndex e_undirected_H;
  SamplesIndex h_directed_E;
  SamplesIndex h_negative_B;
  // assign h_out for vertices to be minimum of outgoing half-edge indices
  // assign v_origin/f_left/h_next for half-edges in H0
  // assign h_bound for faces
  for (Index f = 0; f < Nf; ++f) {
    h_right_F[f] = 3 * f;
    for (Index i = 0; i < 3; ++i) {
      Index h = 3 * f + i;
      Index h_next = 3 * f + (i + 1) % 3;
      Index v0 = V_cycle_F(f, i);
      Index v1 = V_cycle_F(f, (i + 1) % 3);
      H0.set_row(h, {v0, v1});
      v_origin_H[h] = v0;
      f_left_H[h] = f;
      h_next_H[h] = h_next;
      // assign h_out for vertices if not already assigned
      // reassign if h is smaller than current h_out_V[v0]
      if (h_out_V[v0] > h) {
        h_out_V[v0] = h;
      }
    }
  }
  // Temporary containers for indices of +/- boundary half-edge
  std::vector<Index> H_boundary_plus;
  std::unordered_set<Index> H_boundary_minus;
  // find positive boundary half-edges
  // assign h_twin for interior half-edges
  for (Index h = 0; h < H0.rows(); ++h) {
    // if h_twin_H[h] is already assigned, skip
    if (h_twin_H[h] != InvalidIndex) {
      continue;
    }
    Index h_twin = find_halfedge_index_of_twin(H0, h);
    if (h_twin == InvalidIndex) {
      H_boundary_plus.push_back(h);
    } else {
      h_twin_H[h] = h_twin;
      h_twin_H[h_twin] = h;
    }
  }
  Index Nh1 = H_boundary_plus.size();
  Index Nh = Nh0 + Nh1;
  v_origin_H.conservativeResize(Nh);
  h_next_H.conservativeResize(Nh);
  h_twin_H.conservativeResize(Nh);
  f_left_H.conservativeResize(Nh);
  // define negative boundary half-edges
  // assign v_origin for negative boundary half-edges
  // assign h_twin for boundary half-edges
  for (Index i = 0; i < Nh1; ++i) {
    Index h = H_boundary_plus[i];
    Index h_twin = Nh0 + i;
    // Index v0 = H0(h, 0);
    Index v1 = H0(h, 1);
    H_boundary_minus.insert(h_twin);
    v_origin_H[h_twin] = v1;
    h_twin_H[h] = h_twin;
    h_twin_H[h_twin] = h;
  }
  // enumerate boundaries b=0,1,...
  // assign h_right for boundaries
  // assign h_next for negative boundary half-edges
  // set f_left=-(b+1) for half-edges in boundary b
  while (!H_boundary_minus.empty()) {
    Index b = h_negative_B.size();
    Index h_negative_b = *H_boundary_minus.begin();
    h_negative_B.conservativeResize(b + 1);
    h_negative_B[b] = h_negative_b; // Assign new value
    Index h = h_negative_b;
    // follow prev cycle along boundary b until we get back to h=h_negative_b
    do {
      Index h_prev = h_twin_H[h];
      // rotate cw around origin of h until we find h_prev in boundary b
      // erase h from H_boundary_minus
      while (H_boundary_minus.find(h_prev) == H_boundary_minus.end()) {
        h_prev = h_twin_H[h_next_H[h_prev]];
      }
      h_next_H[h_prev] = h;
      h = h_prev;
      H_boundary_minus.erase(h);
      f_left_H[h] = -(b + 1);
    } while (h != h_negative_b);
  }

  // Find undirected edge data
  size_t Ne = Nh / 2;
  e_undirected_H.resize(Nh);
  h_directed_E.resize(Ne);
  // Hmin contains min(h, twin(h)) for all half-edges h
  // tracks whether the pair {h, twin(h)} has been seen
  std::unordered_set<Index> Hmin;
  Hmin.reserve(Ne);
  // Look at half-edges of each face
  // Assign new edge if min(h, twin(h)) has not been seen
  Index e{0};
  for (Index f{0}; f < Nf; ++f) {
    Index h0 = h_right_F[f];
    Index h1 = h_next_H[h0];
    Index h2 = h_next_H[h1];

    // Follow next cycle along face boundary
    Index h = h0;
    Index h_start = h;
    do {
      Index ht = h_twin_H[h];
      Index hn = h_next_H[h];
      Index v = v_origin_H[h];
      Index hmin = std::min(h, ht);

      if (Hmin.find(hmin) == Hmin.end()) {
        // Index e = Hmin.size();
        Index vt = v_origin_H[ht];
        if (vt < v) {
          std::swap(v, vt);
        }
        e_undirected_H[h] = e;
        e_undirected_H[ht] = e;
        h_directed_E[e] = h;

        Hmin.insert(hmin);
        ++e;
      }
      h = hn;
    } while (h != h_start);
  }

  std::map<std::string, SamplesIndex> data_map;

  data_map["h_out_V"] = h_out_V;
  data_map["h_directed_E"] = h_directed_E;
  data_map["h_right_F"] = h_right_F;
  data_map["h_negative_B"] = h_negative_B;

  data_map["v_origin_H"] = v_origin_H;
  data_map["e_undirected_H"] = e_undirected_H;
  data_map["f_left_H"] = f_left_H;

  data_map["h_next_H"] = h_next_H;
  data_map["h_twin_H"] = h_twin_H;

  return data_map;
}

std::map<std::string, SamplesIndex>
half_edge_samples_no_edge_data_to_edge_tri_cycles(
    const std::map<std::string, SamplesIndex> &he_samples) {

  const SamplesIndex &h_out_V = he_samples.at("h_out_V");
  const SamplesIndex &v_origin_H = he_samples.at("v_origin_H");
  const SamplesIndex &h_next_H = he_samples.at("h_next_H");
  const SamplesIndex &h_twin_H = he_samples.at("h_twin_H");
  const SamplesIndex &f_left_H = he_samples.at("f_left_H");
  const SamplesIndex &h_right_F = he_samples.at("h_right_F");

  size_t num_vertices = h_out_V.size();
  size_t num_edges = v_origin_H.size() / 2;
  size_t num_faces = h_right_F.size();
  size_t num_half_edges = v_origin_H.size();

  SamplesIndex V_cycle_E(num_edges, 2);
  SamplesIndex V_cycle_F(num_edges, 3);

  SamplesIndex e_undirected_H(num_half_edges);
  SamplesIndex h_directed_E(num_edges);

  // Hmin contains min(h, twin(h)) for all half-edges h
  // tracks whether the pair {h, twin(h)} has been seen
  std::unordered_set<Index> Hmin;
  Hmin.reserve(num_edges);
  // Look at half-edges of each face
  // Assign new edge if min(h, twin(h)) has not been seen
  Index e{0};
  for (Index f{0}; f < num_faces; ++f) {
    Index h0 = h_right_F[f];
    Index h1 = h_next_H[h0];
    Index h2 = h_next_H[h1];
    V_cycle_F.set_row(f, {v_origin_H[h0], v_origin_H[h1], v_origin_H[h2]});

    // Follow next cycle along face boundary
    Index h = h0;
    Index h_start = h;
    do {
      Index ht = h_twin_H[h];
      Index hn = h_next_H[h];
      Index v = v_origin_H[h];
      Index hmin = std::min(h, ht);

      if (Hmin.find(hmin) == Hmin.end()) {
        // Index e = Hmin.size();
        Index vt = v_origin_H[ht];
        if (vt < v) {
          std::swap(v, vt);
        }
        V_cycle_E.set_row(e, {v, vt});
        e_undirected_H[h] = e;
        e_undirected_H[ht] = e;
        h_directed_E[e] = h;

        Hmin.insert(hmin);
        ++e;
      }
      h = hn;
    } while (h != h_start);
  }

  std::map<std::string, SamplesIndex> data_map;
  data_map["V_cycle_E"] = V_cycle_E;
  data_map["V_cycle_F"] = V_cycle_F;
  return data_map;
}

std::map<std::string, SamplesIndex> half_edge_samples_to_edge_tri_cycles(
    const std::map<std::string, SamplesIndex> &he_samples) {

  const SamplesIndex &h_directed_E = he_samples.at("h_directed_E");
  const SamplesIndex &h_right_F = he_samples.at("h_right_F");

  const SamplesIndex &v_origin_H = he_samples.at("v_origin_H");

  const SamplesIndex &h_next_H = he_samples.at("h_next_H");
  const SamplesIndex &h_twin_H = he_samples.at("h_twin_H");

  size_t num_edges = v_origin_H.size() / 2;
  size_t num_faces = h_right_F.size();

  SamplesIndex V_cycle_E(num_edges, 2);
  SamplesIndex V_cycle_F(num_faces, 3);

  for (Index f{0}; f < num_faces; ++f) {
    Index h0 = h_right_F[f];
    Index h1 = h_next_H[h0];
    Index h2 = h_next_H[h1];
    V_cycle_F.set_row(f, {v_origin_H[h0], v_origin_H[h1], v_origin_H[h2]});
  }

  for (Index e{0}; e < num_edges; ++e) {
    Index h0 = h_directed_E[e];
    Index h1 = h_twin_H[h0];
    V_cycle_E.set_row(e, {v_origin_H[h0], v_origin_H[h1]});
  }

  std::map<std::string, SamplesIndex> data_map;
  data_map["V_cycle_E"] = V_cycle_E;
  data_map["V_cycle_F"] = V_cycle_F;
  return data_map;
}

SimplicialTopology2
HalfEdgeTopology_to_SimplicialTopology2(const HalfEdgeTopology &he_topo) {

  size_t num_vertices = he_topo.num_vertices();
  size_t num_faces = he_topo.num_faces();
  size_t num_edges = he_topo.num_edges();
  size_t num_half_edges = he_topo.num_half_edges();
  size_t num_boundaries = he_topo.num_boundaries();

  SimplicialTopology2 s_topo(num_edges, num_faces);

  SamplesIndex e_undirected_H_(num_half_edges);
  SamplesIndex h_directed_E_(num_edges);

  std::unordered_set<Index> Hmin;
  Hmin.reserve(num_edges);
  for (Index f{0}; f < num_faces; ++f) {
    Index h0 = he_topo.h_right_f(f);
    Index h1 = he_topo.h_next_h(h0);
    Index h2 = he_topo.h_next_h(h1);
    s_topo.V_cycle_F.set_row(f, {he_topo.v_origin_h(h0), he_topo.v_origin_h(h1),
                                 he_topo.v_origin_h(h2)});

    Index h = h0;
    Index h_start = h;
    do {
      Index ht = he_topo.h_twin_h(h);
      Index hn = he_topo.h_next_h(h);
      Index v = he_topo.v_origin_h(h);
      Index hmin = std::min(h, ht);

      if (Hmin.find(hmin) == Hmin.end()) {
        Index e = Hmin.size();
        Index vt = he_topo.v_origin_h(ht);
        if (vt < v) {
          std::swap(v, vt);
        }
        s_topo.V_cycle_E.set_row(e, {v, vt});
        e_undirected_H_[h] = e;
        e_undirected_H_[ht] = e;
        h_directed_E_[e] = h;

        Hmin.insert(hmin);
      }
      h = hn;
    } while (h != h_start);
  }
  return s_topo;
}

//////////////////////////////////////////////
/////////////////////////////////////////////
////////////////////////////////////////////
/////////////////////////////////////////////

// std::map<std::string, SamplesIndex>
// tri_cycles_to_dart_samples(const SamplesIndex &V_cycle_F) {
//
//   using Simplex0 = std::array<Index, 1>;
//   using Simplex1 = std::array<Index, 2>;
//   using Simplex2 = std::array<Index, 3>;
//   using Dart = std::tuple<Simplex0, Simplex1, Simplex2>;
//   using Simplex0Hash = mathutils::hash::ArrayHash<Index, 1>;
//   using Simplex1Hash = mathutils::hash::ArrayHash<Index, 2>;
//   using Simplex2Hash = mathutils::hash::ArrayHash<Index, 3>;
//   struct DartHash {
//     std::size_t operator()(Dart const &d) const noexcept {
//       std::size_t seed = 0;
//       mathutils::hash::hash_combine(seed, Simplex0Hash{}(std::get<0>(d)));
//       mathutils::hash::hash_combine(seed, Simplex1Hash{}(std::get<1>(d)));
//       mathutils::hash::hash_combine(seed, Simplex2Hash{}(std::get<2>(d)));
//       return seed;
//     }
//   };
//
//   std::unordered_set<Simplex0, Simplex0Hash> S0;
//   std::unordered_set<Simplex1, Simplex1Hash> S1;
//   std::unordered_set<Simplex2, Simplex2Hash> S2;
//   std::unordered_set<Dart, DartHash> D;
//
//   std::unordered_map<Simplex0, Dart, Simplex0Hash> d_through_S0;
//   std::unordered_map<Simplex1, Dart, Simplex1Hash> d_through_S1;
//   std::unordered_map<Simplex2, Dart, Simplex2Hash> d_through_S2;
//
//   std::unordered_map<Dart, Simplex0, DartHash> s0_in_D;
//   std::unordered_map<Dart, Simplex1, DartHash> s1_in_D;
//   std::unordered_map<Dart, Simplex2, DartHash> s2_in_D;
//
//   std::unordered_map<Dart, Dart, DartHash> d_cmap0_D;
//   std::unordered_map<Dart, Dart, DartHash> d_cmap1_D;
//   std::unordered_map<Dart, Dart, DartHash> d_cmap2_D;
//
//   std::map<std::string, SamplesIndex> data_map;
//
//   Index Nv = V_cycle_F.maxCoeff() + 1;
//   Index Nf = V_cycle_F.rows();
//
//   std::unordered_map<Dart, int, DartHash> id_S0;
//   std::unordered_map<Dart, int, DartHash> id_S1;
//   std::unordered_map<Dart, int, DartHash> id_S2;
//   std::unordered_map<Dart, int, DartHash> id_D;
//
//   std::unordered_map<Dart, int, DartHash> count_S0;
//   std::unordered_map<Dart, int, DartHash> count_S1;
//   std::unordered_map<Dart, int, DartHash> count_S2;
//
//   struct DART {
//     Index s0, s1, s2, d0, d1, d2;
//   };
//
//   // enumerate
//   for (Index f{0}; f < Nf; ++f) {
//     Index i = V_cycle_F(f, 0);
//     Index j = V_cycle_F(f, 1);
//     Index k = V_cycle_F(f, 2);
//     int p = 1;
//
//     if (j < i) {
//       std::swap(i, j);
//       p = -p;
//     }
//     if (k < i) {
//       std::swap(i, k);
//       p = -p;
//     }
//     if (k < j) {
//       std::swap(j, k);
//       p = -p;
//     }
//
//     // Simplex0 vi({i}), vj({j}), vk({k});
//     Simplex1 eij({i, j}), ejk({j, k}), eik({i, k});
//     Simplex2 fijk({i, j, k});
//   }
//
//   return data_map;
// }

} // namespace mesh
} // namespace mathutils
