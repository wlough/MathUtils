#pragma once

/**
 * @file half_edge_mesh.hpp
 * @brief Simple half-edge mesh class
 */
#include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/mesh_plyio.hpp"
#include "mathutils/random/random.hpp"
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

class HalfEdgeTopology {
  using Generatori = mathutils::SimpleGenerator<Index>;

public:
  ///////////////////////////////
  // Core data structure ////////
  ///////////////////////////////
  SamplesIndex h_out_V;        //
  SamplesIndex h_directed_E;   //
  SamplesIndex h_right_F;      //
  SamplesIndex h_negative_B;   //
                               //
  SamplesIndex v_origin_H;     //
  SamplesIndex e_undirected_H; //
  SamplesIndex f_left_H;       //
                               //
  SamplesIndex h_next_H;       //
  SamplesIndex h_twin_H;       //
  ///////////////////////////////

  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////
  Index h_out_v(Index v) const { return h_out_V[v]; }
  Index h_directed_e(Index e) const { return h_directed_E[e]; }
  Index h_right_f(Index f) const { return h_right_F[f]; }
  Index h_negative_b(Index b) const { return h_negative_B[b]; }

  Index v_origin_h(Index h) const { return v_origin_H[h]; }
  Index e_undirected_h(Index h) const { return e_undirected_H[h]; }
  Index f_left_h(Index h) const { return f_left_H[h]; }

  Index h_next_h(Index h) const { return h_next_H[h]; } // (v', e', f, c)
  Index h_twin_h(Index h) const { return h_twin_H[h]; } // (v', e, f', c)

  Index b_ghost_f(Index f) const { return -f - 1; }

  Index h_in_v(Index v) const { return h_twin_H[h_out_V[v]]; }
  Index v_head_h(Index h) const { return v_origin_H[h_twin_H[h]]; }
  Index h_prev_h(Index h) const; // TODO test this
  Index h_rotcw_h(Index h) const { return h_next_H[h_twin_H[h]]; }
  // Index h_rotccw_h(Index h) const;
  Index h_prev_h_by_rot(Index h) const; // TODO test this
  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  bool some_negative_boundary_contains_h(Index h) const {
    return f_left_h(h) < 0;
  }
  bool some_positive_boundary_contains_h(Index h) const {
    return some_negative_boundary_contains_h(h_twin_h(h));
  }
  bool some_boundary_contains_h(Index h) const {
    return some_negative_boundary_contains_h(h) ||
           some_positive_boundary_contains_h(h);
  }
  bool some_boundary_contains_v(Index v) const {
    return some_boundary_contains_h(h_out_v(v));
  }
  bool h_is_flippable(Index h) const;
  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  Generatori generate_H_outcw_v(Index v, Index h_start = InvalidIndex) const {
    if (h_start == InvalidIndex) {
      h_start = h_out_V[v];
    }
    for (auto h : generate_H_rotcw_h(h_start)) {
      co_yield h;
    }
  }
  Generatori generate_H_right_f(Index f) const {
    Index h_start = h_right_F[f];
    Index h = h_start;
    do {
      co_yield h;
      h = h_next_h(h);
    } while (h != h_start);
  }
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
  Generatori generate_F_incident_v(Index v) const {

    for (auto h : generate_H_rotcw_h(h_out_V[v])) {
      co_yield f_left_H[h];
    }
  }

  ///////////////////////////////////////////
  // Miscellaneous properties ///////////////
  ///////////////////////////////////////////
  /**
   * @brief Get the number of vertices in the mesh
   *
   * @return Index
   */
  Index num_vertices() const { return h_out_V.size(); }
  /**
   * @brief Get the number edges in the mesh
   *
   * @return Index
   */
  Index num_edges() const { return v_origin_H.size() / 2; }
  /**
   * @brief Get the number of faces in the mesh
   *
   * @return Index
   */
  Index num_faces() const { return h_right_F.size(); }

  // Index num_cells() const { return h_above_C_.size(); }

  /**
   * @brief Get the numberhalf edges in the mesh
   *
   * @return Index
   */
  Index num_half_edges() const { return v_origin_H.size(); }
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
  Index num_boundaries() const { return h_negative_B.size(); }
  /**
   * @brief Get the genus of the mesh
   *
   * @return Index
   */
  Index genus() const {
    return (2 - euler_characteristic() - num_boundaries()) / 2;
  }

  MeshSamples to_mesh_samples() const;
  std::map<std::string, SamplesIndex> to_topo_samples() const;

  void from_mesh_samples(MeshSamples &ms);

  /**
   * @brief Flips half-edge h.
   *
   * @param h
   *
   *```
   *         v1                           v1
   *       /    \                       /  |  \
   *      /      \                     /   |   \
   *     /h3    h2\                   /h3  |  h2\
   *    /    f0    \                 /     |     \
   *   /            \               /  f0  |  f1  \
   *  /     h=h0     \             /       |       \
   * v2--------------v0  |----->  v2     h0|h1     v0
   *  \      h1      /             \       |       /
   *   \            /               \      |      /
   *    \    f1    /                 \     |     /
   *     \h4    h5/                   \h4  |  h5/
   *      \      /                     \   |   /
   *       \    /                       \  |  /
   *         v3                           v3
   * ```
   */
  bool flip_hedge(Index h);

  std::vector<SamplesIndex> VB_cycles() const;
};

class HalfEdgeMesh {
private:
  random::RandomNumberGenerator rng_;

public:
  SamplesReal X_ambient_V;
  SamplesIndex V_cycle_E;
  SamplesIndex V_cycle_F;
  HalfEdgeTopology topo;
  MeshSamples attrs;

  std::span<Real> X_ambient_v(Index v) { return X_ambient_V.row_span(v); }
  void set_X_ambient_v(Index v, std::initializer_list<Real> X) {
    X_ambient_V.set_row(v, X);
  }

  MeshSamples to_mesh_samples() const;
  void from_mesh_samples(MeshSamples &ms);
  void load_ply(const std::string &filepath,
                const bool preload_into_memory = true,
                const bool verbose = false,
                const std::string &ply_property_convention = "MathUtils") {
    MeshSamples ms = mesh::io::load_mesh_samples(
        filepath, preload_into_memory, verbose, ply_property_convention);

    from_mesh_samples(ms);
  }

  void save_ply(const std::string &filepath, const bool use_binary = true,
                const std::string &ply_property_convention = "MathUtils") {

    mathutils::mesh::io::save_mesh_samples(to_mesh_samples(), filepath,
                                           use_binary, ply_property_convention);
  }

  /**
   * @brief Add V_cycle_E and V_cycle_F to attrs.
   */
  void refresh_simplex_cycles_from_topo();

  bool h_is_locally_delaunay(Index h) const;

  int flip_non_delaunay();

  void build_icososphere(size_t num_refinements) {
    // MeshSamples ms = build_icososphere_samples(num_refinements);
  }

  /**
   * @brief Divides pair of face by adding a new vertex at midpoint of their
   * shared edge
   *
   * @param f
   * @details
   * ```
   * On Boundary:
   *                 v2                                    v2
   *               /   \                                 / | \
   *              /     \                               /  |  \
   *             /       \                             /   |   \
   *            /         \                           /    |    \
   *           /           \                         /     |     \
   *          /             \                       /      |      \
   *         /               \                     /       |       \
   *        /h2             h1\                   /h2      |      h1\
   *       /        f0         \                 /         |e2       \
   *      /                     \               /        h4|h5        \
   *     /                       \             /    f0     |    f1     \
   *    /                         \           /            |            \
   *   /            e0             \         /    e0       |      e1     \
   *  /             h0              \ ----> /     h0       |      h3      \
   * v0 ----------------------------v1     v0 ------------v3--------------v1
   *  \            ht0              /       \    ht3             ht0      /
   *   \                           /         \                           /
   *    \ht1    ft0=-b0-1      ht2/           \ht1    ft0=-b0-1      ht2/
   *
   * existing half-edges: h0, h1, h2|, ht0, ht1, ht2
   * existing vertices: v0, v1, v2
   * existing edges: e0
   * existing faces: f0, ft0=-b0-1
   *
   * new half-edges: h3, h4, h5|, ht3
   * new vertices: v3
   * new edges: e1, e2
   * new faces: f1
   *
   *
   * Not on boundary:
   *                 v2                                    v2
   *               /   \                                 / | \
   *              /     \                               /  |  \
   *             /       \                             /   |   \
   *            /         \                           /    |    \
   *           /           \                         /     |     \
   *          /             \                       /      |      \
   *         /               \                     /       |       \
   *        /h2             h1\                   /h2      |      h1\
   *       /        f0         \                 /         |e2       \
   *      /                     \               /        h4|h5        \
   *     /                       \             /    f0     |    f1     \
   *    /                         \           /            |            \
   *   /            e0             \         /    e0       |      e1     \
   *  /             h0              \ ----> /     h0       |      h3      \
   * v0 ----------------------------v1     v0 ------------v3--------------v1
   *  \            ht0              /       \    ht3       |     ht0      /
   *   \                           /         \             |             /
   *    \                         /           \            |            /
   *     \                       /             \    ft1    |    ft0    /
   *      \                     /               \       ht5|ht4       /
   *       \       ft0         /                 \         |et2      /
   *        \ht1           ht2/                   \ht1     |     ht2/
   *         \               /                     \       |       /
   *          \             /                       \      |      /
   *           \           /                         \     |     /
   *            \         /                           \    |    /
   *             \       /                             \   |   /
   *              \     /                               \  |  /
   *               \   /                                 \ | /
   *                vt2                                   vt2
   *
   * existing half-edges: h0, h1, h2|, ht0, ht1, ht2
   * existing vertices: v0, v1, v2|, vt2
   * existing edges: e0
   * existing faces: f0|, ft0
   *
   * new half-edges: h3, h4, h5|, ht3, ht4, ht5
   * new vertices: v3
   * new edges: e1, e2|, et2
   * new faces: f1|, ft1
   *
   * ```
   */
  void split_edge(Index e);
};
/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils
