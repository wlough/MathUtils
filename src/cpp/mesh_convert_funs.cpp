/**
 * @file mesh.cpp
 */
#include "mathutils/mesh/mesh_convert_funs.hpp"
#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/simplicial_complex2.hpp"
#include <map>
#include <unordered_set>
// #include <array>
// #include <cstddef>
// #include <tuple>
// #include <unordered_map>

namespace mathutils {
namespace mesh {

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

HalfEdgeTopology
SimplicialTopology2_to_HalfEdgeTopology(const SimplicialTopology2 &s_topo) {
  size_t num_vertices = s_topo.f_incident_V.rows();
  size_t num_faces = s_topo.V_cycle_F.rows();
  size_t num_edges = s_topo.V_cycle_E.rows();

  std::map<std::string, SamplesIndex> ms =
      tri_cycles_to_half_edge_samples(s_topo.V_cycle_F);
  size_t num_boundaries = ms.at("h_negative_B").rows();

  HalfEdgeTopology he_topo(num_vertices, num_edges, num_faces, num_boundaries);
  he_topo.from_topo_samples(ms);

  return he_topo;
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
