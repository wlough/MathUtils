#pragma once

/**
 * @file mesh_common_data_types.hpp
 */

#include "mathutils/simple_generator.hpp"
#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include <cstdint>
#include <map>
#include <string>
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

using MeshSamples = std::map<
    std::string,
    std::variant<
        Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1, Eigen::RowMajor>,
        Eigen::Matrix<std::int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>,
        Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>,
        Eigen::Matrix<std::int32_t, Eigen::Dynamic, 4, Eigen::RowMajor>,
        Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1, Eigen::RowMajor>,
        Eigen::Matrix<std::int64_t, Eigen::Dynamic, 2, Eigen::RowMajor>,
        Eigen::Matrix<std::int64_t, Eigen::Dynamic, 3, Eigen::RowMajor>,
        Eigen::Matrix<std::int64_t, Eigen::Dynamic, 4, Eigen::RowMajor>,
        Eigen::VectorXd,
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
        Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>,
        std::vector<std::vector<std::int32_t>>,
        std::vector<std::vector<std::int64_t>>>>;

template <typename IntType> struct CellComplexTemplate {
  std::vector<std::vector<IntType>> V_cycle_E;
  std::vector<std::vector<IntType>> V_cycle_F;
  std::vector<std::vector<IntType>> V_cycle_C;
};

template <typename IntType> struct SimplicialComplexTemplate {
  using Samplesi = Eigen::Matrix<IntType, Eigen::Dynamic, 1>;
  using Samples2i = Eigen::Matrix<IntType, Eigen::Dynamic, 2>;
  using Samples3i = Eigen::Matrix<IntType, Eigen::Dynamic, 3>;
  Samplesi V_cycle_E;
  Samples2i V_cycle_F;
  Samples3i V_cycle_C;
};

template <typename IntType> struct HalfEdgeMapTemplate {
  using Index = Eigen::Index;
  using Samples5i = Eigen::Matrix<IntType, Eigen::Dynamic, 5, Eigen::RowMajor>;
  using Samplesi = Eigen::Matrix<IntType, Eigen::Dynamic, 1>;
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

  Generatori MatrixMesh::generate_H_next_h(Index h) const {
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

  SimplicialComplexTemplate<IntType> to_simplicial_complex() const {
    SimplicialComplexTemplate<IntType> sc;
    sc.V_cycle_E.resize(h_directed_E.size(), 2);
    for (Index e = 0; e < h_directed_E.size(); ++e) {
      IntType h = h_directed_E(e);
      sc.V_cycle_E(e, 0) = v_origin_h(h);
      sc.V_cycle_E(e, 1) = v_origin_h(h_twin_h(h));
    }
    sc.V_cycle_F.resize(h_right_F.size(), 3);
    for (Index f = 0; f < h_right_F.size(); ++f) {
      IntType h = h_right_F(f);
      for (Index i = 0; i < 3; ++i) {
        sc.V_cycle_F(f, i) = v_origin_h(h);
        h = h_next_h(h);
      }
    }
    return sc;
  }
};

using HalfEdgeMapInt32 = HalfEdgeMapTemplate<std::int32_t>;
using HalfEdgeMapInt64 = HalfEdgeMapTemplate<std::int64_t>;

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

} // namespace mesh
} // namespace mathutils
