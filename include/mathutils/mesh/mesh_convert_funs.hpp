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

#include "mathutils/hash.hpp" // mathutils::hash::ArrayHash, mathutils::hash::hash_combine
#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include <array>
#include <cstddef>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

/////////////////////////////////////
/////////////////////////////////////
// Mesh utilities //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {
using ArrSimplex0 = std::array<int, 1>;
using ArrSimplex1 = std::array<int, 2>;
using ArrSimplex2 = std::array<int, 3>;
using TupDart2 = std::tuple<ArrSimplex0, ArrSimplex1, ArrSimplex2>;

struct TupDart2Hash {
  std::size_t operator()(TupDart2 const &d) const noexcept {
    std::size_t seed = 0;
    mathutils::hash::hash_combine(
        seed, mathutils::hash::ArrayHash<int, 1>{}(std::get<0>(d)));
    mathutils::hash::hash_combine(
        seed, mathutils::hash::ArrayHash<int, 2>{}(std::get<1>(d)));
    mathutils::hash::hash_combine(
        seed, mathutils::hash::ArrayHash<int, 3>{}(std::get<2>(d)));
    return seed;
  }
};

using Samplesi = Eigen::VectorXi;
using Samples2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;
using Samples3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;
using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using Samples2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;

using SetofArrSimplex0 = std::unordered_set<ArrSimplex0>;
using SetofArrSimplex1 = std::unordered_set<ArrSimplex1>;
using SetofArrSimplex2 = std::unordered_set<ArrSimplex2>;
using SetofTupDart2 = std::unordered_set<TupDart2>;

// using TupDart2MapTupDart2 =
//     std::unordered_map<TupDart2, TupDart2, mathutils::hash::ArrayHash<int,
//     6>>;

struct HalfEdgeData {
  Eigen::VectorXi h_out_V;
  Eigen::VectorXi h_directed_E;
  Eigen::VectorXi h_right_F;
  Eigen::VectorXi h_negative_B;

  Eigen::VectorXi v_origin_H;
  Eigen::VectorXi e_undirected_H;
  Eigen::VectorXi f_left_H;

  Eigen::VectorXi h_next_H;
  Eigen::VectorXi h_twin_H;
};

struct HalfFaceData {
  Eigen::VectorXi h_out_V;
  Eigen::VectorXi h_directed_E;
  Eigen::VectorXi h_right_F;
  Eigen::VectorXi h_above_C;
  Eigen::VectorXi h_negative_B;

  Eigen::VectorXi v_origin_H;
  Eigen::VectorXi e_undirected_H;
  Eigen::VectorXi f_left_H;
  Eigen::VectorXi c_below_H;

  Eigen::VectorXi h_next_H;
  Eigen::VectorXi h_twin_H;
  Eigen::VectorXi h_flip_H;
};

struct CombinatorialMap2Data {
  Eigen::VectorXi d_through_S0;
  Eigen::VectorXi d_through_S1;
  Eigen::VectorXi d_through_S2;

  Eigen::VectorXi s0_in_D;
  Eigen::VectorXi s1_in_D;
  Eigen::VectorXi s2_in_D;

  Eigen::VectorXi d_cmap0_D;
  Eigen::VectorXi d_cmap1_D;
  Eigen::VectorXi d_cmap2_D;
};

struct DartComplex2Data {

  using Simplex0 = std::array<int, 1>;
  using Simplex1 = std::array<int, 2>;
  using Simplex2 = std::array<int, 3>;
  using Dart = std::tuple<Simplex0, Simplex1, Simplex2>;
  using Simplex0Hash = mathutils::hash::ArrayHash<int, 1>;
  using Simplex1Hash = mathutils::hash::ArrayHash<int, 2>;
  using Simplex2Hash = mathutils::hash::ArrayHash<int, 3>;
  using DartHash = mathutils::mesh::TupDart2Hash;

  std::unordered_set<Simplex0, Simplex0Hash> simplices0;
  std::unordered_set<Simplex1, Simplex1Hash> simplices1;
  std::unordered_set<Simplex2, Simplex2Hash> simplices2;
  std::unordered_set<Dart, DartHash> darts;

  std::unordered_map<Simplex0, Dart, Simplex0Hash> dart_through_Simplex0;
  std::unordered_map<Simplex1, Dart, Simplex1Hash> dart_through_Simplex1;
  std::unordered_map<Simplex2, Dart, Simplex2Hash> dart_through_Simplex2;

  std::unordered_map<Dart, Simplex0, Simplex0Hash> simplex0_in_Dart;
  std::unordered_map<Dart, Simplex1, Simplex1Hash> simplex1_in_Dart;
  std::unordered_map<Dart, Simplex2, Simplex2Hash> simplex2_in_Dart;

  std::unordered_map<Dart, Dart, DartHash> dart_cmap0_Dart;
  std::unordered_map<Dart, Dart, DartHash> dart_cmap1_Dart;
  std::unordered_map<Dart, Dart, DartHash> dart_cmap2_Dart;
};

/**
 * @brief Find the index ht of H[ht]=[j, i] in H, where H[h]=[i, j]. Return -1
 * if not found.
 *
 * @param H Half-edge samples (Nh, 2)
 * @param h Index of half-edge in H
 * @return Index of twin half-edge in H, or -1 if not found
 */
int find_halfedge_index_of_twin(const Samples2i &H, const int &h);

/**
 * @brief Convert triangle vertex cycles to half-edge samples.
 *
 * @param V_cycle_F (Nf, 3) triangle vertex cycles
 * @return std::map<std::string, Samplesi> Half-edge samples and related data
 */
std::map<std::string, Samplesi>
tri_vertex_cycles_to_half_edge_samples(const Samples3i &V_cycle_F);

} // namespace mesh
} // namespace mathutils
