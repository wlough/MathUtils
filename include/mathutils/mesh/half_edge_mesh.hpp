#pragma once

/**
 * @file half_edge_mesh.hpp
 * @brief Simple half-edge mesh class
 */
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/simple_generator.hpp"
// #include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
// #include <array>
// #include <cstddef>
// #include <tuple>
// #include <unordered_set>

// #include <unordered_map>

namespace mathutils {
namespace mesh {
/** @addtogroup Mesh
 *  @{
 */

class SimplicialTopology2 {
public:
  SamplesIndex V_cycle_E_;
  SamplesIndex V_cycle_F_;

  SimplicialTopology2() = default;

  SamplesIndex &V_cycle_E() { return V_cycle_E_; }
  SamplesIndex &V_cycle_F() { return V_cycle_F_; }

  std::span<Index> V_cycle_e(Index e) { return V_cycle_E_.row(e); }
  std::span<Index> V_cycle_f(Index f) { return V_cycle_F_.row(f); }
};

class HalfEdgeTopology {
  using Generatori = mathutils::SimpleGenerator<Index>;

public:
  ////////////////////////////
  // Core data structure /////
  ////////////////////////////
  SamplesIndex h_out_V_;
  SamplesIndex h_directed_E_;
  SamplesIndex h_right_F_;
  SamplesIndex h_negative_B_;

  SamplesIndex v_origin_H_;
  SamplesIndex e_undirected_H_;
  SamplesIndex f_left_H_;

  SamplesIndex h_next_H_; // (v, e, f, c) --> (v', e', f, c)
  SamplesIndex h_twin_H_; // (v, e, f, c) --> (v', e, f', c)
  ////////////////////////////

  SamplesIndex &h_out_V() { return h_out_V_; }
  SamplesIndex &h_directed_E() { return h_directed_E_; }
  SamplesIndex &h_right_F() { return h_right_F_; }
  // SamplesIndex &h_above_C() { return h_above_C_; }
  SamplesIndex &h_negative_B() { return h_negative_B_; }

  SamplesIndex &v_origin_H() { return v_origin_H_; }
  SamplesIndex &e_undirected_H() { return e_undirected_H_; }
  SamplesIndex &f_left_H() { return f_left_H_; }

  SamplesIndex &h_next_H() { return h_next_H_; }
  SamplesIndex &h_twin_H() { return h_twin_H_; }

  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////
  Index h_out_v(Index v) const { return h_out_V_[v]; }
  Index h_directed_e(Index e) const { return h_directed_E_[e]; }
  Index h_right_f(Index f) const { return h_right_F_[f]; }
  // Index h_above_c(Index c) const { return h_above_C_[c]; }
  Index h_negative_b(Index b) const { return h_negative_B_[b]; }

  Index v_origin_h(Index h) const { return v_origin_H_[h]; }
  Index e_undirected_h(Index h) const { return e_undirected_H_[h]; }
  Index f_left_h(Index h) const { return f_left_H_[h]; }

  Index h_next_h(Index h) const { return h_next_H_[h]; } // (v', e', f, c)
  Index h_twin_h(Index h) const { return h_twin_H_[h]; } // (v', e, f', c)

  Index b_ghost_f(Index f) const { return -f - 1; }

  Index h_in_v(Index v) const { return h_twin_H_[h_out_V_[v]]; }
  Index v_head_h(Index h) const { return v_origin_H_[h_twin_H_[h]]; }
  Index h_prev_h(Index h) const { return h_next_H_[h_next_H_[h]]; }
  Index h_rotcw_h(Index h) const;
  Index h_rotccw_h(Index h) const;
  Index h_prev_h_by_rot(Index h) const;

  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  bool some_negative_boundary_contains_h(Index h) const {
    return (f_left_h(h) < 0);
  }
  bool some_positive_boundary_contains_h(Index h) const {
    return some_negative_boundary_contains_h(h_twin_h(h));
  }
  bool some_boundary_contains_h(Index h) const {
    return some_boundary_contains_h(h) ||
           some_negative_boundary_contains_h(h_twin_h(h));
  }
  bool some_boundary_contains_v(Index v) const {
    return some_boundary_contains_h(h_out_v(v));
  }
  bool h_is_locally_delaunay(Index h) const;
  bool h_is_flippable(Index h) const { return !some_boundary_contains_h(h); }
  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  Generatori generate_H_out_v_clockwise(Index v, Index h_start = -1) const;
  Generatori generate_H_right_f(Index f) const;
  Generatori generate_H_rotcw_h(Index h) const {
    Index h_start = h;
    do {
      co_yield h;
      h = h_rotcw_h(h);
    } while (h != h_start);
  }
  Generatori generate_H_next_h(Index h) const {
    Index h_start = h;
    do {
      co_yield h;
      h = h_next_h(h);
    } while (h != h_start);
  }
  Generatori generate_H_right_b(Index b) const {
    Index h_start = h_negative_b(b);
    Index h = h_start;
    do {
      co_yield h;
      h = h_next_h(h);
    } while (h != h_start);
  }
  Generatori generate_F_incident_v(Index v) const;

  ///////////////////////////////////////////
  // Miscellaneous properties ///////////////
  ///////////////////////////////////////////
  /**
   * @brief Get the number of vertices in the mesh
   *
   * @return Index
   */
  Index num_vertices() const { return h_out_V_.size(); }
  /**
   * @brief Get the number edges in the mesh
   *
   * @return Index
   */
  Index num_edges() const { return v_origin_H_.size() / 2; }
  /**
   * @brief Get the number of faces in the mesh
   *
   * @return Index
   */
  Index num_faces() const { return h_right_F_.size(); }

  // Index num_cells() const { return h_above_C_.size(); }

  /**
   * @brief Get the numberhalf edges in the mesh
   *
   * @return Index
   */
  Index num_half_edges() const { return v_origin_H_.size(); }
  /**
   * @brief Get the Euler characteristic of the mesh
   *
   * @return Index
   */
  int euler_characteristic() const {
    return num_vertices() - num_edges() + num_faces();
  }
  /**
   * @brief Get the number of dicconnected boundary components
   *
   * @return Index
   */
  Index num_boundaries() const { return h_negative_B_.size(); }
  /**
   * @brief Get the genus of the mesh
   *
   * @return Index
   */
  Index genus() const {
    return (2 - euler_characteristic() - num_boundaries()) / 2;
  }

  MeshSamples to_mesh_samples() const;

  void from_mesh_samples(const MeshSamples &ms);
};

class HalfEdgeMesh {

public:
  SamplesReal X_ambient_V_;
  HalfEdgeTopology topo;

  SamplesReal &X_ambient_V() { return X_ambient_V_; }
  std::span<Real> X_ambient_v(Index v) { return X_ambient_V_.row(v); }

  MeshSamples to_mesh_samples() const;
  void from_mesh_samples(const MeshSamples &ms);
};
/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils
