#pragma once

/**
 * @file mesh_common_data_types.hpp
 */

#include "mathutils/simple_generator.hpp"
#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <variant>
#include <vector> // std::vector
// #include <array>
// #include <cstddef>
// #include <map>
// #include <tuple>
// #include <unordered_map>
// #include <unordered_set>

/////////////////////////////////////
/////////////////////////////////////
// Mesh data types //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {

// template <typename IntType, std::size_t Dim>
// using SamplesNiTemplate =
//     Eigen::Matrix<IntType, Eigen::Dynamic, Dim, Eigen::RowMajor>;
template <typename IntType, int N>
using SamplesNiTemplate =
    Eigen::Matrix<IntType, Eigen::Dynamic, N,
                  (N == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
template <typename IntType>
using RaggedSamplesiTemplate = std::vector<std::vector<IntType>>;

using Samplesi32 = SamplesNiTemplate<std::int32_t, 1>;
using Samples2i32 = SamplesNiTemplate<std::int32_t, 2>;
using Samples3i32 = SamplesNiTemplate<std::int32_t, 3>;
using Samples4i32 = SamplesNiTemplate<std::int32_t, 4>;
using Samples5i32 = SamplesNiTemplate<std::int32_t, 5>;
using Samples6i32 = SamplesNiTemplate<std::int32_t, 6>;
using Samples7i32 = SamplesNiTemplate<std::int32_t, 7>;
using Samples8i32 = SamplesNiTemplate<std::int32_t, 8>;

using Samplesi64 = SamplesNiTemplate<std::int64_t, 1>;
using Samples2i64 = SamplesNiTemplate<std::int64_t, 2>;
using Samples3i64 = SamplesNiTemplate<std::int64_t, 3>;
using Samples4i64 = SamplesNiTemplate<std::int64_t, 4>;
using Samples5i64 = SamplesNiTemplate<std::int64_t, 5>;
using Samples6i64 = SamplesNiTemplate<std::int64_t, 6>;
using Samples7i64 = SamplesNiTemplate<std::int64_t, 7>;
using Samples8i64 = SamplesNiTemplate<std::int64_t, 8>;

using Samplesi = SamplesNiTemplate<int, 1>;
using Samples2i = SamplesNiTemplate<int, 2>;
using Samples3i = SamplesNiTemplate<int, 3>;
using Samples4i = SamplesNiTemplate<int, 4>;
using Samples5i = SamplesNiTemplate<int, 5>;
using Samples6i = SamplesNiTemplate<int, 6>;
using Samples7i = SamplesNiTemplate<int, 7>;
using Samples8i = SamplesNiTemplate<int, 8>;

using RaggedSamplesi32 = RaggedSamplesiTemplate<std::int32_t>;
using RaggedSamplesi64 = RaggedSamplesiTemplate<std::int64_t>;
using RaggedSamplesi = RaggedSamplesiTemplate<int>;
using Samplesd = Eigen::VectorXd;
using Samples2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Samples4d = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

template <typename IntType>
using MeshSamplesTemplate = std::map<
    std::string,
    std::variant<SamplesNiTemplate<IntType, 1>, SamplesNiTemplate<IntType, 2>,
                 SamplesNiTemplate<IntType, 3>, SamplesNiTemplate<IntType, 4>,
                 SamplesNiTemplate<IntType, 5>, SamplesNiTemplate<IntType, 6>,
                 SamplesNiTemplate<IntType, 7>, SamplesNiTemplate<IntType, 8>,
                 Samplesd, Samples2d, Samples3d, Samples4d,
                 std::vector<std::vector<IntType>>>>;

using MeshSamples = MeshSamplesTemplate<int>;
// using MeshSamples =
//     std::map<std::string,
//              std::variant<Samplesi, Samples2i, Samples3i, Samples4i,
//              Samples5i,
//                           Samples6i, Samples7i, Samples8i, Samplesd,
//                           Samples2d, Samples3d, Samples4d,
//                           std::vector<std::vector<int>>>>;
using MeshSamples32 = MeshSamplesTemplate<std::int32_t>;
using MeshSamples64 = MeshSamplesTemplate<std::int64_t>;

// using MeshSamplesMixed = std::map<
//     std::string,
//     std::variant<Samplesi32, Samples2i32, Samples3i32, Samples4i32,
//     Samples5i32,
//                  Samples6i32, Samples7i32, Samples8i32, Samplesi64,
//                  Samples2i64, Samples3i64, Samples4i64, Samples5i64,
//                  Samples6i64, Samples7i64, Samples8i64, Samplesi, Samples2i,
//                  Samples3i, Samples4i, Samples5i, Samples6i, Samples7i,
//                  Samples8i, Samplesd, Samples2d, Samples3d, Samples4d,
//                  RaggedSamplesi32, RaggedSamplesi64>>;

// /**
//  * @brief Check if int -> int32 cast would overflow, and perform the cast.
//  *
//  * @param v
//  * @return * template <typename Mat64>
//  */
// template <typename Mat64>
// static auto checked_cast_eigen_int_to_i32(const Mat64 &v);
// /**
//  * @brief Check if int-> int32 cast would overflow, and perform the cast.
//  *
//  * @param in
//  * @return  std::vector<std::int32_t>
//  */
// static std::vector<std::int32_t>
// checked_cast_vec_int_to_i32(const std::vector<int> &in);

/**
 * @brief Convert MeshSamples to MeshSamples32 by casting integer types.
 */
MeshSamples32 convert_mesh_samples_to_32(const MeshSamples &mesh_samples);

// template <typename IntType> struct CellComplexTemplate {
//   RaggedSamplesiTemplate<IntType> V_cycle_E;
//   RaggedSamplesiTemplate<IntType> V_cycle_F;
//   RaggedSamplesiTemplate<IntType> V_cycle_C;
// };

// template <typename IntType> struct SimplicialComplexTemplate {
//   SamplesiTemplate<IntType> V_cycle_E;
//   Samples2iTemplate<IntType> V_cycle_F;
//   Samples3iTemplate<IntType> V_cycle_C;
// };

// struct HalfFaceData {
//   Eigen::VectorXi h_out_V;
//   Eigen::VectorXi h_directed_E;
//   Eigen::VectorXi h_right_F;
//   Eigen::VectorXi h_above_C;
//   Eigen::VectorXi h_negative_B;

//   Eigen::VectorXi v_origin_H;
//   Eigen::VectorXi e_undirected_H;
//   Eigen::VectorXi f_left_H;
//   Eigen::VectorXi c_below_H;

//   Eigen::VectorXi h_next_H;
//   Eigen::VectorXi h_twin_H;
//   Eigen::VectorXi h_flip_H;
// };

// struct CombinatorialMap2Data {
//   Eigen::VectorXi d_through_S0;
//   Eigen::VectorXi d_through_S1;
//   Eigen::VectorXi d_through_S2;

//   Eigen::VectorXi s0_in_D;
//   Eigen::VectorXi s1_in_D;
//   Eigen::VectorXi s2_in_D;

//   Eigen::VectorXi d_cmap0_D;
//   Eigen::VectorXi d_cmap1_D;
//   Eigen::VectorXi d_cmap2_D;
// };

template <typename IntType> struct HalfEdgeMapTemplate {
  using Index = Eigen::Index;
  using Samples5i = Eigen::Matrix<IntType, Eigen::Dynamic, 5, Eigen::RowMajor>;
  using Samplesi = Eigen::Matrix<IntType, Eigen::Dynamic, 1, Eigen::RowMajor>;
  using Generatori = mathutils::GeneratorTemplate<IntType>;

  /////////////////////////
  // Core data structure //
  /////////////////////////
  /**
   * @brief Matrix vefnt_H[h] = [v_origin, e_undirected, f_left, h_next, h_twin]
   */
  Samples5i vefnt_H;
  Samplesi h_out_V, h_directed_E, h_right_F, h_negative_B;

  ///////////////////////////////////////////
  // Fundamental accessors and properties ///
  ///////////////////////////////////////////
  IntType h_out_v(Index v) const { return h_out_V(v); }
  IntType h_directed_e(Index e) const { return h_directed_E(e); }
  IntType h_right_f(Index f) const { return h_right_F(f); }
  IntType h_negative_b(Index b) const { return h_negative_B(b); }

  IntType v_origin_h(Index h) const { return vefnt_H(h, 0); }
  IntType e_undirected_h(Index h) const { return vefnt_H(h, 1); }
  IntType f_left_h(Index h) const { return vefnt_H(h, 2); }

  IntType h_next_h(Index h) const { return vefnt_H(h, 3); }
  IntType h_twin_h(Index h) const { return vefnt_H(h, 4); }

  IntType v_head_h(Index h) const { return vefnt_H(vefnt_H(h, 4), 0); }
  IntType h_rotcw_h(Index h) const { return vefnt_H(vefnt_H(h, 4), 3); }

  Index num_vertices() const { return h_out_V.size(); }
  Index num_edges() const { return vefnt_H.rows() / 2; }
  Index num_faces() const { return h_right_F.size(); }
  Index num_boundaries() const { return h_negative_B.size(); }
  Index num_half_edges() const { return vefnt_H.rows(); }
  IntType euler_characteristic() const {
    return num_vertices() - num_edges() + num_faces();
  }

  bool some_negative_boundary_contains_h(Index h) const {
    return f_left_h(h) < 0;
  }
  IntType b_ghost_f(IntType f) const { return -f - 1; }

  Generatori generate_H_next_h(Index h) const {
    int h_start = h;
    do {
      co_yield h;
      h = h_next_h(h);
    } while (h != h_start);
  }

  Generatori generate_H_rotcw_h(Index h) const {
    int h_start = h;
    do {
      co_yield h;
      h = h_rotcw_h(h);
    } while (h != h_start);
  }

  MeshSamplesTemplate<IntType> to_mesh_samples() const {
    MeshSamplesTemplate<IntType> ms;
    ms["h_out_V"] = h_out_V;
    ms["h_directed_E"] = h_directed_E;
    ms["h_right_F"] = h_right_F;
    ms["h_negative_B"] = h_negative_B;
    ms["v_origin_H"] = vefnt_H.col(0);
    ms["e_undirected_H"] = vefnt_H.col(1);
    ms["f_left_H"] = vefnt_H.col(2);
    ms["h_next_H"] = vefnt_H.col(3);
    ms["h_twin_H"] = vefnt_H.col(4);
    return ms;
  }

  // SimplicialComplexTemplate<IntType> to_simplicial_complex() const {
  //   SimplicialComplexTemplate<IntType> sc;
  //   sc.V_cycle_E.resize(h_directed_E.size(), 2);
  //   for (Index e = 0; e < h_directed_E.size(); ++e) {
  //     IntType h = h_directed_E(e);
  //     sc.V_cycle_E(e, 0) = v_origin_h(h);
  //     sc.V_cycle_E(e, 1) = v_origin_h(h_twin_h(h));
  //   }
  //   sc.V_cycle_F.resize(h_right_F.size(), 3);
  //   for (Index f = 0; f < h_right_F.size(); ++f) {
  //     IntType h = h_right_F(f);
  //     for (Index i = 0; i < 3; ++i) {
  //       sc.V_cycle_F(f, i) = v_origin_h(h);
  //       h = h_next_h(h);
  //     }
  //   }
  //   return sc;
  // }
};

using HalfEdgeMap = HalfEdgeMapTemplate<int>;
using HalfEdgeMap32 = HalfEdgeMapTemplate<std::int32_t>;
using HalfEdgeMap64 = HalfEdgeMapTemplate<std::int64_t>;

/**
 * @brief Template struct for a vertex set where template parameter is dimension
 * of the space in which the vertices live.
 *
 */
template <int Dim> struct VertexSetTemplate {
  using SamplesNd = Eigen::Matrix<double, Eigen::Dynamic, Dim, Eigen::RowMajor>;
  using RowCoordsNd = Eigen::Matrix<double, 1, Dim, Eigen::RowMajor>;
  using Index = Eigen::Index;

  SamplesNd coord_V;

  static constexpr int dimension() { return Dim; }
  Index num_vertices() const { return coord_V.rows(); }

  RowCoordsNd coord_v(Index v) const { return coord_V.row(v).eval(); }
};

using VertexSetDim2 = VertexSetTemplate<2>;
using VertexSetDim3 = VertexSetTemplate<3>;

} // namespace mesh
} // namespace mathutils
