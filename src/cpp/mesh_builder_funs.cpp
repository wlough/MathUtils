/**
 * @file mesh_builder_funs.cpp
 */

#include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/hash.hpp" // mathutils::hash::ArrayHash, mathutils::hash::hash_combine
#include "mathutils/matrix.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

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
  // Nh0 = num interior + num positive boundary half-edges
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
  SamplesIndex V_cycle_F(num_faces, 3);

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

std::map<std::string, SamplesIndex> half_edge_samples_to_simplicial_samples(
    const std::map<std::string, SamplesIndex> &he_samples) {

  const SamplesIndex &h_out_V = he_samples.at("h_out_V");
  const SamplesIndex &h_directed_E = he_samples.at("h_directed_E");
  const SamplesIndex &h_right_F = he_samples.at("h_right_F");

  const SamplesIndex &v_origin_H = he_samples.at("v_origin_H");
  const SamplesIndex &e_undirected_H = he_samples.at("e_undirected_H");
  const SamplesIndex &f_left_H = he_samples.at("f_left_H");

  const SamplesIndex &h_next_H = he_samples.at("h_next_H");
  const SamplesIndex &h_twin_H = he_samples.at("h_twin_H");

  size_t num_vertices = h_out_V.size();
  size_t num_edges = v_origin_H.size() / 2;
  size_t num_faces = h_right_F.size();

  SamplesIndex V_cycle_E(num_edges, 2);
  SamplesIndex V_cycle_F(num_faces, 3);
  SamplesIndex f_incident_V(num_vertices);
  SamplesIndex F_incident_E(num_edges, 2);
  SamplesIndex E_incident_F(num_faces, 3);

  for (Index f{0}; f < num_faces; ++f) {
    Index h0 = h_right_F[f];
    Index h1 = h_next_H[h0];
    Index h2 = h_next_H[h1];
    Index e0 = e_undirected_H[h0];
    Index e1 = e_undirected_H[h1];
    Index e2 = e_undirected_H[h2];
    V_cycle_F.set_row(f, {v_origin_H[h0], v_origin_H[h1], v_origin_H[h2]});
    E_incident_F.set_row(
        f, {e_undirected_H[h0], e_undirected_H[h1], e_undirected_H[h2]});
  }

  for (Index e{0}; e < num_edges; ++e) {
    Index h0 = h_directed_E[e];
    Index h1 = h_twin_H[h0];
    V_cycle_E.set_row(e, {v_origin_H[h0], v_origin_H[h1]});
    F_incident_E.set_row(e, {f_left_H[h0], f_left_H[h1]});
  }

  for (Index v{0}; v < num_vertices; ++v) {
    Index h = h_out_V[v];
    f_incident_V[v] = f_left_H[h];
  }

  std::map<std::string, SamplesIndex> data_map;
  data_map["V_cycle_E"] = V_cycle_E;
  data_map["V_cycle_F"] = V_cycle_F;
  data_map["f_incident_V"] = f_incident_V;
  data_map["F_incident_E"] = F_incident_E;
  data_map["E_incident_F"] = E_incident_F;
  return data_map;
}

MeshSamples build_icosohedron_simplicial_samples() {
  MeshSamples ms;
  double phi = (1.0 + sqrt(5.0)) * 0.5; // golden ratio 1.61803...
  double a = 1.0;
  double b = 1.0 / phi;

  int num_vertices = 12;
  int num_edges = 30;
  int num_faces = 20;
  Matrix<Real> X_ambient_V(num_vertices, 3);
  Matrix<Index> V_cycle_E(num_edges, 2);
  Matrix<Index> V_cycle_F(num_faces, 3);

  X_ambient_V.set_row(0, {0.0, b, -a});
  X_ambient_V.set_row(1, {b, a, 0.0});
  X_ambient_V.set_row(2, {-b, a, 0.0});
  X_ambient_V.set_row(3, {0.0, b, a});
  X_ambient_V.set_row(4, {0.0, -b, a});
  X_ambient_V.set_row(5, {-a, 0.0, b});
  X_ambient_V.set_row(6, {0.0, -b, -a});
  X_ambient_V.set_row(7, {a, 0.0, -b});
  X_ambient_V.set_row(8, {a, 0.0, b});
  X_ambient_V.set_row(9, {-a, 0.0, -b});
  X_ambient_V.set_row(10, {b, -a, 0.0});
  X_ambient_V.set_row(11, {-b, -a, 0.0});

  double rad = std::sqrt(a * a + b * b);
  X_ambient_V *= 1.0 / rad;

  V_cycle_F.set_row(0, {2, 1, 0});
  V_cycle_F.set_row(1, {1, 2, 3});
  V_cycle_F.set_row(2, {5, 4, 3});
  V_cycle_F.set_row(3, {4, 8, 3});
  V_cycle_F.set_row(4, {7, 6, 0});
  V_cycle_F.set_row(5, {6, 9, 0});
  V_cycle_F.set_row(6, {11, 10, 4});
  V_cycle_F.set_row(7, {10, 11, 6});
  V_cycle_F.set_row(8, {9, 5, 2});
  V_cycle_F.set_row(9, {5, 9, 11});
  V_cycle_F.set_row(10, {8, 7, 1});
  V_cycle_F.set_row(11, {7, 8, 10});
  V_cycle_F.set_row(12, {2, 5, 3});
  V_cycle_F.set_row(13, {8, 1, 3});
  V_cycle_F.set_row(14, {9, 2, 0});
  V_cycle_F.set_row(15, {1, 7, 0});
  V_cycle_F.set_row(16, {11, 9, 6});
  V_cycle_F.set_row(17, {7, 10, 6});
  V_cycle_F.set_row(18, {5, 11, 4});
  V_cycle_F.set_row(19, {10, 8, 4});

  // {2, 1, 0} -> {1, 2}, {0, 1}, {0, 2}
  // {1, 2, 3} -> {1, 2}, {2, 3}, {1, 3}
  // {5, 4, 3} -> {4, 5}, {3, 4}, {3, 5}
  // {4, 8, 3} -> {4, 8}, {3, 8}, {3, 4}
  // {7, 6, 0} -> {6, 7}, {0, 6}, {0, 7}
  // {6, 9, 0} -> {6, 9}, {0, 9}, {0, 6}
  // {11, 10, 4} -> {10, 11}, {4, 10}, {4, 11}
  // {10, 11, 6} -> {10, 11}, {6, 11}, {6, 10}
  // {9, 5, 2} -> {5, 9}, {2, 5}, {2, 9}
  // {5, 9, 11} -> {5, 9}, {9, 11}, {5, 11}
  // {8, 7, 1} -> {7, 8}, {1, 7}, {1, 8}
  // {7, 8, 10} -> {7, 8}, {8, 10}, {7, 10}
  // {2, 5, 3} -> {2, 5}, {3, 5}, {2, 3}
  // {8, 1, 3} -> {1, 8}, {1, 3}, {3, 8}
  // {9, 2, 0} -> {2, 9}, {0, 2}, {0, 9}
  // {1, 7, 0} -> {1, 7}, {0, 7}, {0, 1}
  // {11, 9, 6} -> {9, 11}, {6, 9}, {6, 11}
  // {7, 10, 6} -> {7, 10}, {6, 10}, {6, 7}
  // {5, 11, 4} -> {5, 11}, {4, 11}, {4, 5}
  // {10, 8, 4} -> {8, 10}, {4, 8}, {4, 10}

  // {0, 1}, {0, 2}, {1, 2}
  // {1, 2}, {1, 3}, {2, 3}
  // {3, 4}, {3, 5}, {4, 5}
  // {3, 4}, {3, 8}, {4, 8}
  // {0, 6}, {0, 7}, {6, 7}
  // {0, 6}, {0, 9}, {6, 9}
  // {4, 10}, {4, 11}, {10, 11}
  // {6, 10}, {6, 11}, {10, 11}
  // {2, 5}, {2, 9}, {5, 9}
  // {5, 9}, {5, 11}, {9, 11}
  // {1, 7}, {1, 8}, {7, 8}
  // {7, 8}, {7, 10}, {8, 10}
  // {2, 3}, {2, 5}, {3, 5}
  // {1, 3}, {1, 8}, {3, 8}
  // {0, 2}, {0, 9}, {2, 9}
  // {0, 1}, {0, 7}, {1, 7}
  // {6, 9}, {6, 11}, {9, 11}
  // {6, 7}, {6, 10}, {7, 10}
  // {4, 5}, {4, 11}, {5, 11}
  // {4, 8}, {4, 10}, {8, 10}

  V_cycle_E.set_row(0, {0, 1});
  V_cycle_E.set_row(1, {0, 2});
  V_cycle_E.set_row(2, {0, 6});
  V_cycle_E.set_row(3, {0, 7});
  V_cycle_E.set_row(4, {0, 9});
  V_cycle_E.set_row(5, {1, 2});
  V_cycle_E.set_row(6, {1, 3});
  V_cycle_E.set_row(7, {1, 7});
  V_cycle_E.set_row(8, {1, 8});
  V_cycle_E.set_row(9, {2, 3});
  V_cycle_E.set_row(10, {2, 5});
  V_cycle_E.set_row(11, {2, 9});
  V_cycle_E.set_row(12, {3, 4});
  V_cycle_E.set_row(13, {3, 5});
  V_cycle_E.set_row(14, {3, 8});
  V_cycle_E.set_row(15, {4, 5});
  V_cycle_E.set_row(16, {4, 8});
  V_cycle_E.set_row(17, {4, 10});
  V_cycle_E.set_row(18, {4, 11});
  V_cycle_E.set_row(19, {5, 9});
  V_cycle_E.set_row(20, {5, 11});
  V_cycle_E.set_row(21, {6, 7});
  V_cycle_E.set_row(22, {6, 9});
  V_cycle_E.set_row(23, {6, 10});
  V_cycle_E.set_row(24, {6, 11});
  V_cycle_E.set_row(25, {7, 8});
  V_cycle_E.set_row(26, {7, 10});
  V_cycle_E.set_row(27, {8, 10});
  V_cycle_E.set_row(28, {9, 11});
  V_cycle_E.set_row(29, {10, 11});

  ms["X_ambient_V"] = X_ambient_V;
  ms["V_cycle_E"] = V_cycle_E;
  ms["V_cycle_F"] = V_cycle_F;
  return ms;
}

void refine_vertex_face_samples(MeshSamples &ms) {

  auto vertex_pair_key = [](Index v0, Index v1) -> std::uint64_t {
    std::uint32_t a = static_cast<std::uint32_t>(std::min(v0, v1));
    std::uint32_t b = static_cast<std::uint32_t>(std::max(v0, v1));
    return (std::uint64_t(a) << 32) | std::uint64_t(b);
  };

  SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
  SamplesIndex &V_cycle_F0 = std::get<SamplesIndex>(ms.at("V_cycle_F"));
  // SamplesIndex &V_cycle_E0 = std::get<SamplesIndex>(ms.at("V_cycle_E"));

  Index num_faces0 = V_cycle_F0.rows();
  Index num_faces = 4 * num_faces0;
  Index num_vertices0 = X_ambient_V.rows();

  SamplesIndex V_cycle_F(num_faces, 3);

  std::vector<SamplesReal> newV;
  newV.reserve(num_vertices0 +
               num_faces0); // num_vertices0 + num_faces0 - 2 for sphere

  std::unordered_map<std::uint64_t, Index> v_midpt_vv;
  Index v_count = num_vertices0;
  Index f_count = 0;

  printf("for (Index f = 0; f < num_faces0; f++)");
  for (Index f = 0; f < num_faces0; f++) {
    Index v0 = V_cycle_F0(f, 0);
    Index v1 = V_cycle_F0(f, 1);
    Index v2 = V_cycle_F0(f, 2);
    long long key01 = vertex_pair_key(v0, v1);
    long long key12 = vertex_pair_key(v1, v2);
    long long key20 = vertex_pair_key(v2, v0);

    Index v01 =
        (v_midpt_vv.count(key01) == 1) ? v_midpt_vv[key01] : InvalidIndex;
    Index v12 =
        (v_midpt_vv.count(key12) == 1) ? v_midpt_vv[key12] : InvalidIndex;
    Index v20 =
        (v_midpt_vv.count(key20) == 1) ? v_midpt_vv[key20] : InvalidIndex;

    if (v01 == InvalidIndex) {
      v01 = v_count++;
      SamplesReal xyz01 =
          (X_ambient_V.row_copy(v0) + X_ambient_V.row_copy(v1)) / 2.0;
      newV.push_back(xyz01);
      v_midpt_vv[key01] = v01;
    }
    if (v12 == InvalidIndex) {
      v12 = v_count++;
      SamplesReal xyz12 =
          (X_ambient_V.row_copy(v1) + X_ambient_V.row_copy(v2)) / 2.0;
      newV.push_back(xyz12);
      v_midpt_vv[key12] = v12;
    }
    if (v20 == InvalidIndex) {
      v20 = v_count++;
      SamplesReal xyz20 =
          (X_ambient_V.row_copy(v2) + X_ambient_V.row_copy(v0)) / 2.0;
      newV.push_back(xyz20);
      v_midpt_vv[key20] = v20;
    }

    V_cycle_F.set_row(f_count++, {v0, v01, v20});
    V_cycle_F.set_row(f_count++, {v01, v1, v12});
    V_cycle_F.set_row(f_count++, {v20, v12, v2});
    V_cycle_F.set_row(f_count++, {v01, v12, v20});
  }

  X_ambient_V.conservativeResize(num_vertices0 + newV.size(), 3);
  printf("for (Index v = 0; v < newV.size(); v++)");
  for (Index v = 0; v < newV.size(); v++) {
    X_ambient_V.set_row(v + num_vertices0,
                        {newV[v][0], newV[v][1], newV[v][2]});
  }

  ms["X_ambient_V"] = X_ambient_V;
  ms["V_cycle_F"] = V_cycle_F;
}

void refine_simplicial_samples(MeshSamples &ms) {

  auto vertex_pair_key = [](Index v0, Index v1) -> std::uint64_t {
    std::uint32_t a = static_cast<std::uint32_t>(std::min(v0, v1));
    std::uint32_t b = static_cast<std::uint32_t>(std::max(v0, v1));
    return (std::uint64_t(a) << 32) | std::uint64_t(b);
  };

  SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
  SamplesIndex &V_cycle_E0 = std::get<SamplesIndex>(ms.at("V_cycle_E"));
  SamplesIndex &V_cycle_F0 = std::get<SamplesIndex>(ms.at("V_cycle_F"));

  size_t Nv0 = X_ambient_V.rows();
  size_t Ne0 = V_cycle_E0.rows();
  size_t Nf0 = V_cycle_F0.rows();

  size_t Nv = Nv0 + Ne0;
  size_t Ne = 2 * Ne0 + 3 * Nf0;
  size_t Nf = 4 * Nf0;

  X_ambient_V.conservativeResize(Nv, 3);
  SamplesIndex V_cycle_E(Ne, 2);
  SamplesIndex V_cycle_F(Nf, 3);

  std::unordered_map<std::uint64_t, Index> v_midpt_vv;
  Index v_count = Nv0;
  Index f_count = 0;
  Index e_count = 0;

  // printf("for (Index f = 0; f < num_faces0; f++)");
  for (Index f = 0; f < Nf0; f++) {
    Index v0 = V_cycle_F0(f, 0);
    Index v1 = V_cycle_F0(f, 1);
    Index v2 = V_cycle_F0(f, 2);
    long long key01 = vertex_pair_key(v0, v1);
    long long key12 = vertex_pair_key(v1, v2);
    long long key20 = vertex_pair_key(v2, v0);

    Index v01 =
        (v_midpt_vv.count(key01) == 1) ? v_midpt_vv[key01] : InvalidIndex;
    Index v12 =
        (v_midpt_vv.count(key12) == 1) ? v_midpt_vv[key12] : InvalidIndex;
    Index v20 =
        (v_midpt_vv.count(key20) == 1) ? v_midpt_vv[key20] : InvalidIndex;

    if (v01 == InvalidIndex) {
      v01 = v_count++;
      SamplesReal xyz01 =
          (X_ambient_V.row_copy(v0) + X_ambient_V.row_copy(v1)) / 2.0;
      X_ambient_V.set_row(v01, {xyz01[0], xyz01[1], xyz01[2]});
      V_cycle_E.set_row(e_count++, {v0, v01});
      V_cycle_E.set_row(e_count++, {v01, v1});
      v_midpt_vv[key01] = v01;
    }
    if (v12 == InvalidIndex) {
      v12 = v_count++;
      SamplesReal xyz12 =
          (X_ambient_V.row_copy(v1) + X_ambient_V.row_copy(v2)) / 2.0;
      X_ambient_V.set_row(v12, {xyz12[0], xyz12[1], xyz12[2]});
      V_cycle_E.set_row(e_count++, {v1, v12});
      V_cycle_E.set_row(e_count++, {v12, v2});
      v_midpt_vv[key12] = v12;
    }
    if (v20 == InvalidIndex) {
      v20 = v_count++;
      SamplesReal xyz20 =
          (X_ambient_V.row_copy(v2) + X_ambient_V.row_copy(v0)) / 2.0;
      X_ambient_V.set_row(v20, {xyz20[0], xyz20[1], xyz20[2]});
      V_cycle_E.set_row(e_count++, {v2, v20});
      V_cycle_E.set_row(e_count++, {v20, v0});
      v_midpt_vv[key20] = v20;
    }

    V_cycle_E.set_row(e_count++, {v01, v12});
    V_cycle_E.set_row(e_count++, {v12, v20});
    V_cycle_E.set_row(e_count++, {v20, v01});

    V_cycle_F.set_row(f_count++, {v0, v01, v20});
    V_cycle_F.set_row(f_count++, {v01, v1, v12});
    V_cycle_F.set_row(f_count++, {v20, v12, v2});
    V_cycle_F.set_row(f_count++, {v01, v12, v20});
  }

  ms["X_ambient_V"] = X_ambient_V;
  ms["V_cycle_E"] = V_cycle_E;
  ms["V_cycle_F"] = V_cycle_F;
}

MeshSamples build_icososphere_simplicial_samples(size_t num_refinements) {
  MeshSamples ms = build_icosohedron_simplicial_samples();
  SamplesReal &X_ambient_V0 = std::get<SamplesReal>(ms.at("X_ambient_V"));
  Index Nv0 = X_ambient_V0.rows();
  for (size_t refinement; refinement < num_refinements; ++refinement) {
    refine_simplicial_samples(ms);
    SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
    Index Nv = X_ambient_V.rows();
    for (Index v = Nv0; v < Nv; ++v) {
      SamplesReal xyz = X_ambient_V.row_copy(v);
      xyz /= xyz.norm();
      X_ambient_V.set_row(v, {xyz[0], xyz[1], xyz[2]});
    }
    Nv0 = Nv;
  }

  return ms;
}

MeshSamples build_icososphere_half_edge_samples(size_t num_refinements) {
  MeshSamples he_samples;
  MeshSamples s_samples = build_icososphere_simplicial_samples(num_refinements);
  SamplesIndex V_cycle_F;
  assign_matrix_from_variant(s_samples.at("V_cycle_F"), V_cycle_F);

  std::map<std::string, SamplesIndex> ms =
      tri_cycles_to_half_edge_samples(V_cycle_F);

  he_samples["X_ambient_V"] = s_samples.at("X_ambient_V");
  he_samples["V_cycle_E"] = s_samples.at("V_cycle_E");
  he_samples["V_cycle_F"] = s_samples.at("V_cycle_F");

  he_samples["h_out_V"] = ms.at("h_out_V");
  he_samples["h_directed_E"] = ms.at("h_directed_E");
  he_samples["h_right_F"] = ms.at("h_right_F");
  he_samples["h_negative_B"] = ms.at("h_negative_B");

  he_samples["v_origin_H"] = ms.at("v_origin_H");
  he_samples["e_undirected_H"] = ms.at("e_undirected_H");
  he_samples["f_left_H"] = ms.at("f_left_H");

  he_samples["h_next_H"] = ms.at("h_next_H");
  he_samples["h_twin_H"] = ms.at("h_twin_H");
  return he_samples;
}

} // namespace mesh
} // namespace mathutils
