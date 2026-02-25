#pragma once

/**
 * @file mesh.hpp
 * @brief Mesh tools
 * Provides:
 *   - `HalfEdgeData`: simple struct for 2D half-edge mesh
 *   - `HalfFaceData`: simple struct for 3D half-face mesh
 *   - `CombinatorialMap2Data`: simple struct for combinatorial map of a 2D
 * complex
 */

// #include "mathutils/hash.hpp" // mathutils::hash::ArrayHash,
// mathutils::hash::hash_combine
#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/simplicial_complex2.hpp"
// #include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
// #include <array>
// #include <cstddef>
#include <map>
// #include <tuple>
// #include <unordered_map>
// #include <unordered_set>

/////////////////////////////////////
/////////////////////////////////////
// Mesh utilities //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {
// using ArrSimplex0 = std::array<int, 1>;
// using ArrSimplex1 = std::array<int, 2>;
// using ArrSimplex2 = std::array<int, 3>;
// using TupDart2 = std::tuple<ArrSimplex0, ArrSimplex1, ArrSimplex2>;
//
// struct TupDart2Hash {
//   std::size_t operator()(TupDart2 const &d) const noexcept {
//     std::size_t seed = 0;
//     mathutils::hash::hash_combine(
//         seed, mathutils::hash::ArrayHash<int, 1>{}(std::get<0>(d)));
//     mathutils::hash::hash_combine(
//         seed, mathutils::hash::ArrayHash<int, 2>{}(std::get<1>(d)));
//     mathutils::hash::hash_combine(
//         seed, mathutils::hash::ArrayHash<int, 3>{}(std::get<2>(d)));
//     return seed;
//   }
// };

// using Samplesi = Eigen::VectorXi;
// using Samples2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;
// using Samples3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;
// using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;
// using Samples2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;
//
// using SetofArrSimplex0 = std::unordered_set<ArrSimplex0>;
// using SetofArrSimplex1 = std::unordered_set<ArrSimplex1>;
// using SetofArrSimplex2 = std::unordered_set<ArrSimplex2>;
// using SetofTupDart2 = std::unordered_set<TupDart2>;

// using TupDart2MapTupDart2 =
//     std::unordered_map<TupDart2, TupDart2, mathutils::hash::ArrayHash<int,
//     6>>;
//
// struct DartComplex2Data {
//
//   using Simplex0 = std::array<int, 1>;
//   using Simplex1 = std::array<int, 2>;
//   using Simplex2 = std::array<int, 3>;
//   using Dart = std::tuple<Simplex0, Simplex1, Simplex2>;
//   using Simplex0Hash = mathutils::hash::ArrayHash<int, 1>;
//   using Simplex1Hash = mathutils::hash::ArrayHash<int, 2>;
//   using Simplex2Hash = mathutils::hash::ArrayHash<int, 3>;
//   using DartHash = mathutils::mesh::TupDart2Hash;
//
//   std::unordered_set<Simplex0, Simplex0Hash> simplices0;
//   std::unordered_set<Simplex1, Simplex1Hash> simplices1;
//   std::unordered_set<Simplex2, Simplex2Hash> simplices2;
//   std::unordered_set<Dart, DartHash> darts;
//
//   std::unordered_map<Simplex0, Dart, Simplex0Hash> dart_through_Simplex0;
//   std::unordered_map<Simplex1, Dart, Simplex1Hash> dart_through_Simplex1;
//   std::unordered_map<Simplex2, Dart, Simplex2Hash> dart_through_Simplex2;
//
//   std::unordered_map<Dart, Simplex0, Simplex0Hash> simplex0_in_Dart;
//   std::unordered_map<Dart, Simplex1, Simplex1Hash> simplex1_in_Dart;
//   std::unordered_map<Dart, Simplex2, Simplex2Hash> simplex2_in_Dart;
//
//   std::unordered_map<Dart, Dart, DartHash> dart_cmap0_Dart;
//   std::unordered_map<Dart, Dart, DartHash> dart_cmap1_Dart;
//   std::unordered_map<Dart, Dart, DartHash> dart_cmap2_Dart;
// };

/**
 * @brief Find the index ht of H[ht]=[j, i] in H, where H[h]=[i, j]. Return
 * InvalidIndex if not found.
 *
 * @param H Half-edge samples (Nh, 2)
 * @param h Index of half-edge in H
 * @return Index of twin half-edge in H, or InvalidIndex if not found
 */
Index find_halfedge_index_of_twin(const SamplesIndex &H, const Index &h);

/**
 * @brief Convert triangle vertex cycles to edge vertex cycles.
 *
 * @param V_cycle_F triangle vertex cycles (Nh, 3)
 * @return SamplesIndex V_cycle_E edge vertex cycles (Nh, 2)
 */
SamplesIndex tri_cycles_to_edge_cycles(const SamplesIndex &V_cycle_F);

/**
 * @brief Convert triangle vertex cycles to half-edge samples.
 *
 * @param V_cycle_F (Nf, 3) triangle vertex cycles
 * @return std::map<std::string, Samplesi> Half-edge samples and related data
 */
std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples_alt(const SamplesIndex &V_cycle_F);

/**
 * @brief Convert triangle vertex cycles to half-edge samples.
 *
 * @param V_cycle_F (Nf, 3) triangle vertex cycles
 * @return std::map<std::string, SampleSimple half-edge mesh classsi> Half-edge
 * samples and related data
 */
std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples(const SamplesIndex &V_cycle_F);

std::map<std::string, SamplesIndex>
half_edge_samples_no_edge_data_to_edge_tri_cycles(
    const std::map<std::string_view, SamplesIndex> &he_samples);

std::map<std::string_view, SamplesIndex> half_edge_samples_to_edge_tri_cycles(
    const std::map<std::string_view, SamplesIndex> &he_samples);

std::map<std::string, SamplesIndex>
tri_cycles_to_dart_samples(const SamplesIndex &V_cycle_F);

std::map<std::string, SamplesIndex> half_edge_samples_to_dart_samples(
    const std::map<std::string, SamplesIndex> &ms);

SimplicialTopology2
HalfEdgeTopology_to_SimplicialTopology2(const HalfEdgeTopology &he_topo);

HalfEdgeTopology
SimplicialTopology2_to_HalfEdgeTopology(const SimplicialTopology2 &he_topo);
} // namespace mesh
} // namespace mathutils
