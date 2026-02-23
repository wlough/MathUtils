/**
 * @file mesh.cpp
 */
#include "mathutils/mesh/mesh_convert_funs.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include <map>
#include <unordered_set>
#include <vector>
// #include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
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

std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples(const SamplesIndex &V_cycle_F) {

  Index Nv = V_cycle_F.maxCoeff() + 1;
  Index Nf = V_cycle_F.rows();
  // num interior + num positive boundary half-edges
  Index Nh0 = 3 * Nf;
  SamplesIndex H0 = SamplesIndex(Nh0, 2);
  // half-edge samples
  // h_out=Nh0 if not assigned
  // h_twin=InvalidIndex if not assigned
  SamplesIndex h_out_V = SamplesIndex(Nv);
  h_out_V.fill(Nh0);
  SamplesIndex v_origin_H = SamplesIndex(Nh0);
  SamplesIndex h_next_H = SamplesIndex(Nh0);
  SamplesIndex h_twin_H = SamplesIndex(Nh0);
  h_twin_H.fill(InvalidIndex);
  SamplesIndex f_left_H = SamplesIndex(Nh0);
  SamplesIndex h_right_F = SamplesIndex(Nf);
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
  std::map<std::string, SamplesIndex> halfedge_data_map;
  halfedge_data_map["h_out_V"] = h_out_V;
  halfedge_data_map["v_origin_H"] = v_origin_H;
  halfedge_data_map["h_next_H"] = h_next_H;
  halfedge_data_map["h_twin_H"] = h_twin_H;
  halfedge_data_map["f_left_H"] = f_left_H;
  halfedge_data_map["h_right_F"] = h_right_F;
  halfedge_data_map["h_negative_B"] = h_negative_B;
  return halfedge_data_map;
}

SimplicialTopology2
HalfEdgeTopology_to_SimplicialTopology2(const HalfEdgeTopology he_topo) {

  size_t num_vertices = he_topo.num_vertices();
  size_t num_faces = he_topo.num_faces();
  size_t num_edges = he_topo.num_edges();
  size_t num_half_edges = he_topo.num_half_edges();
  size_t num_boundaries = he_topo.num_boundaries();

  SimplicialTopology2 s_topo(num_edges, num_faces);

  SamplesIndex e_undirected_H_(num_half_edges);
  SamplesIndex h_directed_E_(num_edges);

  std::unordered_set<Index> Hmin;
  for (int f{0}; f < num_faces; ++f) {
    Index h0 = he_topo.h_right_f(f);
    Index h1 = he_topo.h_next_h(h0);
    Index h2 = he_topo.h_next_h(h1);
    s_topo.V_cycle_F_.set_row(f,
                              {he_topo.v_origin_h(h0), he_topo.v_origin_h(h1),
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
        // if (v < vt) {
        //   V_cycle_E_.row(e) << v, vt;
        // } else {
        //   V_cycle_E_.row(e) << vt, v;
        // }
        s_topo.V_cycle_E_.set_row(e, {v, vt});
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

std::map<std::string, SamplesIndex>
tri_cycles_to_dart_samples(const SamplesIndex &V_cycle_F) {

  using Simplex0 = std::array<Index, 1>;
  using Simplex1 = std::array<Index, 2>;
  using Simplex2 = std::array<Index, 3>;
  using Dart = std::tuple<Simplex0, Simplex1, Simplex2>;
  using Simplex0Hash = mathutils::hash::ArrayHash<Index, 1>;
  using Simplex1Hash = mathutils::hash::ArrayHash<Index, 2>;
  using Simplex2Hash = mathutils::hash::ArrayHash<Index, 3>;
  struct DartHash {
    std::size_t operator()(Dart const &d) const noexcept {
      std::size_t seed = 0;
      mathutils::hash::hash_combine(seed, Simplex0Hash{}(std::get<0>(d)));
      mathutils::hash::hash_combine(seed, Simplex1Hash{}(std::get<1>(d)));
      mathutils::hash::hash_combine(seed, Simplex2Hash{}(std::get<2>(d)));
      return seed;
    }
  };

  std::unordered_set<Simplex0, Simplex0Hash> S0;
  std::unordered_set<Simplex1, Simplex1Hash> S1;
  std::unordered_set<Simplex2, Simplex2Hash> S2;
  std::unordered_set<Dart, DartHash> D;

  std::unordered_map<Simplex0, Dart, Simplex0Hash> d_through_S0;
  std::unordered_map<Simplex1, Dart, Simplex1Hash> d_through_S1;
  std::unordered_map<Simplex2, Dart, Simplex2Hash> d_through_S2;

  std::unordered_map<Dart, Simplex0, DartHash> s0_in_D;
  std::unordered_map<Dart, Simplex1, DartHash> s1_in_D;
  std::unordered_map<Dart, Simplex2, DartHash> s2_in_D;

  std::unordered_map<Dart, Dart, DartHash> d_cmap0_D;
  std::unordered_map<Dart, Dart, DartHash> d_cmap1_D;
  std::unordered_map<Dart, Dart, DartHash> d_cmap2_D;

  std::map<std::string, SamplesIndex> data_map;

  Index Nv = V_cycle_F.maxCoeff() + 1;
  Index Nf = V_cycle_F.rows();

  std::unordered_map<Dart, int, DartHash> id_S0;
  std::unordered_map<Dart, int, DartHash> id_S1;
  std::unordered_map<Dart, int, DartHash> id_S2;
  std::unordered_map<Dart, int, DartHash> id_D;

  std::unordered_map<Dart, int, DartHash> count_S0;
  std::unordered_map<Dart, int, DartHash> count_S1;
  std::unordered_map<Dart, int, DartHash> count_S2;

  struct DART {
    Index s0, s1, s2, d0, d1, d2;
  };

  // enumerate
  for (Index f{0}; f < Nf; f++) {
    Index i = V_cycle_F(f, 0);
    Index j = V_cycle_F(f, 1);
    Index k = V_cycle_F(f, 2);
    int p = 1;

    if (j < i) {
      std::swap(i, j);
      p = -p;
    }
    if (k < i) {
      std::swap(i, k);
      p = -p;
    }
    if (k < j) {
      std::swap(j, k);
      p = -p;
    }

    // Simplex0 vi({i}), vj({j}), vk({k});
    Simplex1 eij({i, j}), ejk({j, k}), eik({i, k});
    Simplex2 fijk({i, j, k});
  }

  return data_map;
}

} // namespace mesh
} // namespace mathutils
