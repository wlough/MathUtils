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

/////////////////////////////////////
/////////////////////////////////////
// Mesh utilities //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {

class SimplicialComplexBase {
public:
  SamplesField X_ambient_V_; // coords in ambient space
  SamplesIndex V_cycle_E_;
  SamplesIndex V_cycle_F_;
  SamplesIndex V_cycle_C_;

  SimplicialComplexBase() = default;

  SamplesField &X_ambient_V() { return X_ambient_V_; }
  SamplesIndex &V_cycle_E() { return V_cycle_E_; }
  SamplesIndex &V_cycle_F() { return V_cycle_F_; }
  SamplesIndex &V_cycle_C() { return V_cycle_C_; }

  std::span<Real> X_ambient_v(Index v) { return X_ambient_V_.row(v); }
  std::span<Index> V_cycle_e(Index e) { return V_cycle_E_.row(e); }
  std::span<Index> V_cycle_f(Index f) { return V_cycle_F_.row(f); }
  std::span<Index> V_cycle_c(Index c) { return V_cycle_C_.row(c); }
};

class HalfPlexMesh : public SimplicialComplexBase {
  using Generatori = mathutils::SimpleGenerator<Index>;

public:
  ////////////////////////////
  // Core data structure /////
  ////////////////////////////
  SamplesIndex h_out_V_;
  SamplesIndex h_directed_E_;
  SamplesIndex h_right_F_;
  SamplesIndex h_above_C_;
  SamplesIndex h_negative_B_;

  SamplesIndex v_origin_H_;
  SamplesIndex e_undirected_H_;
  SamplesIndex f_left_H_;
  SamplesIndex c_below_H_;

  SamplesIndex h_next_H_;
  SamplesIndex h_twin_H_;
  SamplesIndex h_flip_H_;
  ////////////////////////////

  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////
  Index h_out_v(Index v) const { return h_out_V_[v]; }
  Index h_directed_e(Index e) const { return h_directed_E_[e]; }
  Index h_right_f(Index f) const { return h_right_F_[f]; }
  Index h_above_c(Index c) const { return h_above_C_[c]; }
  Index h_negative_b(Index b) const { return h_negative_B_[b]; }

  Index v_origin_h(Index h) const { return v_origin_H_[h]; }
  Index e_undirected_h(Index h) const { return e_undirected_H_[h]; }
  Index f_left_h(Index h) const { return f_left_H_[h]; }
  Index c_below_h(Index h) const { return c_below_H_[h]; }

  Index h_next_h(Index h) const { return h_next_H_[h]; } // (v', e', f, c)
  Index h_twin_h(Index h) const { return h_twin_H_[h]; } // (v', e, f', c)
  Index h_flip_h(Index h) const { return h_flip_H_[h]; } // (v', e, f, c')

  // Derived combinatorial maps
  // Index h_beta0_h(Index h) const;            // (v', e, f, c)
  // Index h_beta1_h(Index h) const { return; } // (v, e', f, c)
  // Index h_beta2_h(Index h) const;            // (v, e, f', c)
  // Index h_beta3_h(Index h) const;            // (v, e, f, c')

  Index b_ghost_f(Index f) const { return -f - 1; }

  Index h_in_v(Index v) const { return h_out_V_[v]; }
  Index v_head_h(Index h) const;
  Index h_prev_h(Index h) const;
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
  Generatori generate_V_cycle_f(Index f) const;
  Generatori generate_V_of_f(Index f) const;
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
  Generatori generate_H_right_b(Index b) const;
  Generatori generate_F_incident_v(Index v) const;

  ///////////////////////////////////////////
  // Miscellaneous properties ///////////////
  ///////////////////////////////////////////
  /**
   * @brief Get the number of vertices in the mesh
   *
   * @return Index
   */
  Index num_vertices() const;
  /**
   * @brief Get the number edges in the mesh
   *
   * @return Index
   */
  Index num_edges() const;
  /**
   * @brief Get the number of faces in the mesh
   *
   * @return Index
   */
  Index num_faces() const;
  /**
   * @brief Get the number of cells in the mesh
   *
   * @return Index
   */
  Index num_cells() const;
  /**
   * @brief Get the numberhalf edges in the mesh
   *
   * @return Index
   */
  Index num_half_edges() const;
  /**
   * @brief Get the Euler characteristic of the mesh
   *
   * @return Index
   */
  int euler_characteristic() const;
  /**
   * @brief Get the number of dicconnected boundary components
   *
   * @return Index
   */
  Index num_boundaries() const;
  /**
   * @brief Get the genus of the mesh
   *
   * @return Index
   */
  Index genus() const;

  MeshSamples to_mesh_samples() const {
    MeshSamples ms;
    ms["h_out_V"] = h_out_V_;
    ms["h_directed_E"] = h_directed_E_;
    ms["h_right_F"] = h_right_F_;
    ms["h_above_C"] = h_above_C_;
    ms["h_negative_B"] = h_negative_B_;
    ms["v_origin_H"] = v_origin_H_;
    ms["e_undirected_H"] = e_undirected_H_;
    ms["f_left_H"] = f_left_H_;
    ms["c_below_H"] = c_below_H_;
    ms["h_next_H"] = h_next_H_;
    ms["h_twin_H"] = h_twin_H_;
    ms["h_flip_H"] = h_flip_H_;
    return ms;
  }
};


} // namespace mesh
} // namespace mathutils
