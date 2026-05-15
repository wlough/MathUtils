#pragma once

/**
 * @file half_edge_mesh.hpp
 * @brief Simple half-edge mesh class
 */
// #include "mathutils/matrix.hpp"
#include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/mesh_plyio.hpp"
#include "mathutils/random/random.hpp"
#include "mathutils/simple_generator.hpp"

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
  HalfEdgeTopology() = default;
  HalfEdgeTopology(size_t Nv, size_t Ne, size_t Nf, size_t Nb = 0)
      : h_out_V(SamplesIndex(Nv)), h_directed_E(SamplesIndex(Ne)),
        h_right_F(SamplesIndex(Nf)), h_negative_B(SamplesIndex(Nb)),
        v_origin_H(SamplesIndex(2 * Ne)), e_undirected_H(SamplesIndex(2 * Ne)),
        f_left_H(SamplesIndex(2 * Ne)), h_next_H(SamplesIndex(2 * Ne)),
        h_twin_H(SamplesIndex(2 * Ne)) {}

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
  /**
   * \todo test me!
   */
  Index h_prev_h(Index h) const;
  Index h_rotcw_h(Index h) const { return h_next_H[h_twin_H[h]]; }
  Index h_rotccw_h(Index h) const {
    Index h0 = h;
    Index h1 = h;
    do {
      h0 = h1;
      h1 = h_rotcw_h(h1);
    } while (h1 != h);
    return h0;
  }
  /**
   * \todo test me!
   */
  Index h_prev_h_by_rot(Index h) const;
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

  /**
   * ```
   *                 v2                     v2
   *               /   \                      \
   *              /     \                      \
   *             /       \                      \
   *            /         \                      \
   *           /           \                      \
   *          /             \                      \
   *         /               \                      \
   *      h7/h2             h1\h6                  h7\h6
   * f3    / e2     f0      e1 \   f2          f3   e1\    f2
   *      /                     \                      \
   *     /                       \                      \
   *    /                         \                      \
   *   /            e0             \                      \
   *  /             h0              \   ---->              \
   * v0 ----------------------------v1                      v1
   *  \             h3              /                      /
   *   \                           /                      /
   *    \                         /                      /
   *     \                       /                      /
   *      \         f1          /                      /
   *       \                   /                      /
   * f4     \                 /    f5           f4   /   f5
   *       h8\h4           h5/h9                  h8/h9
   *          \e3         e4/                    e3/
   *           \           /                      /
   *            \         /                      /
   *             \       /                      /
   *              \     /                      /
   *               \   /                      /
   *                v3                     vt2
   *```
   *
   * removed
   * h0, h1, h2, h3, h4, h5
   * v0
   * e0, e2, e3
   * f0, f1
   *
   */
  bool h_is_collapsable(Index h) const;

  /**
   *  \todo Finish me
   */
  bool collapse_hedge(Index h);

  /**
   * \todo debug
   */
  void swap_h_indices(Index h0, Index h1) {
    Index h_next_h0 = h_next_H[h0];
    Index h_twin_h0 = h_twin_H[h0];
    Index v_origin_h0 = v_origin_H[h0];
    Index e_undirected_h0 = e_undirected_H[h0];
    Index f_left_h0 = f_left_H[h0];
    Index h_prev_h0 = h_prev_h(h0);

    Index h_next_h1 = h_next_H[h1];
    Index h_twin_h1 = h_twin_H[h1];
    Index v_origin_h1 = v_origin_H[h1];
    Index e_undirected_h1 = e_undirected_H[h1];
    Index f_left_h1 = f_left_H[h1];
    Index h_prev_h1 = h_prev_h(h1);

    // update things that point to h0
    h_next_H[h_prev_h0] = h1;
    h_twin_H[h_twin_h0] = h1;
    if (h_out_V[v_origin_h0] == h0) {
      h_out_V[v_origin_h0] = h1;
    }
    if (h_directed_E[e_undirected_h0] == h0) {
      h_directed_E[e_undirected_h0] = h1;
    }
    if (h_right_F[f_left_h0] == h0) {
      h_right_F[f_left_h0] = h1;
    }

    // update things that point to h1
    h_next_H[h_prev_h1] = h0;
    h_twin_H[h_twin_h1] = h0;
    if (h_out_V[v_origin_h1] == h1) {
      h_out_V[v_origin_h1] = h0;
    }
    if (h_directed_E[e_undirected_h1] == h1) {
      h_directed_E[e_undirected_h1] = h0;
    }
    if (h_right_F[f_left_h1] == h1) {
      h_right_F[f_left_h1] = h0;
    }

    // update things h0 points to
    h_next_H[h0] = h_next_h1;
    h_twin_H[h0] = h_twin_h1;
    v_origin_H[h0] = v_origin_h1;
    e_undirected_H[h0] = e_undirected_h1;
    f_left_H[h0] = f_left_h1;

    // update things h1 points to
    h_next_H[h1] = h_next_h0;
    h_twin_H[h1] = h_twin_h0;
    v_origin_H[h1] = v_origin_h0;
    e_undirected_H[h1] = e_undirected_h0;
    f_left_H[h1] = f_left_h0;
  }

  void swap_v_indices(Index v0, Index v1) {
    Index h0 = h_out_V[v0];
    Index h1 = h_out_V[v1];

    // update things that point to v0
    for (auto h : generate_H_outcw_v(v0)) {
      v_origin_H[h] = v1;
    }
    // update things that point to v1
    for (auto h : generate_H_outcw_v(v1)) {
      v_origin_H[h] = v0;
    }

    // update things v0 points to
    h_out_V[v0] = h1;
    // update things v1 points to
    h_out_V[v1] = h0;
  }
  void swap_e_indices(Index e0, Index e1) {
    Index h0 = h_directed_E[e0];
    Index ht0 = h_twin_H[h0];
    Index h1 = h_directed_E[e1];
    Index ht1 = h_twin_H[h1];

    // update things that point to e0
    e_undirected_H[h0] = e1;
    e_undirected_H[ht0] = e1;
    // update things that point to e1
    e_undirected_H[h1] = e0;
    e_undirected_H[ht1] = e0;
    // update things e0 points to
    h_directed_E[e0] = h1;
    // update things e1 points to
    h_directed_E[e1] = h0;
  };

  void swap_f_indices(Index f0, Index f1) {
    Index h00 = h_right_F[f0];

    Index h10 = h_right_F[f1];

    // update things that point to f0
    for (auto h : generate_H_right_f(f0)) {
      f_left_H[h] = f1;
    }
    // update things that point to f1
    for (auto h : generate_H_right_f(f1)) {
      f_left_H[h] = f0;
    }
    // update things f0 points to
    h_right_F[f0] = h10;
    // update things f1 points to
    h_right_F[f1] = h00;
  };

  // void swap_f_indices(Index f0, Index f1) {
  //   if (f0 == f1)
  //     return;
  //
  //   // Swap representative halfedge pointers
  //   std::swap(h_right_F[f0], h_right_F[f1]);
  //
  //   // Fix halfedge -> face mapping globally
  //   for (Index h = 0; h < num_half_edges(); ++h) {
  //     if (f_left_H[h] == f0)
  //       f_left_H[h] = f1;
  //     else if (f_left_H[h] == f1)
  //       f_left_H[h] = f0;
  //   }
  // }

  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  Generatori generate_H_outcw_v(Index v, Index h_start = InvalidIndex) const {
    if (h_start == InvalidIndex) {
      h_start = h_out_V[v];
    }
    // for (auto h : generate_H_rotcw_h(h_start)) {
    //   co_yield h;
    // }
    Index h = h_start;
    do {
      co_yield h;
      h = h_rotcw_h(h);
    } while (h != h_start);
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

    // for (auto h : generate_H_rotcw_h(h_out_V[v])) {
    //   co_yield f_left_H[h];
    // }
    Index h_start = h_out_V[v];
    Index h = h_start;
    do {
      co_yield f_left_H[h];
      h = h_rotcw_h(h);
    } while (h != h_start);
  }
  Generatori generate_E_incident_v(Index v) const {

    for (auto h : generate_H_rotcw_h(h_out_V[v])) {
      co_yield e_undirected_H[h];
    }
  }
  Generatori generate_V_adjacent_v(Index v) const {

    // for (auto h : generate_H_rotcw_h(h_out_V[v])) {
    //   co_yield v_head_h(h);
    // }
    Index h_start = h_out_V[v];
    Index h = h_start;
    do {
      co_yield v_head_h(h);
      h = h_rotcw_h(h);
    } while (h != h_start);
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

  void from_topo_samples(std::map<std::string, SamplesIndex> ms) {
    h_out_V = ms.at("h_out_V");
    h_directed_E = ms.at("h_directed_E");
    h_right_F = ms.at("h_right_F");
    h_negative_B = ms.at("h_negative_B");

    v_origin_H = ms.at("v_origin_H");
    e_undirected_H = ms.at("e_undirected_H");
    f_left_H = ms.at("f_left_H");

    h_next_H = ms.at("h_next_H");
    h_twin_H = ms.at("h_twin_H");
  }

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

  // void set_attr(std::string key, SamplesVariant value) { attrs[key] = value;
  // };
  void set_attr(const std::string &key, SamplesVariant value) {
    attrs[key] = std::move(value);
  }
  SamplesVariant &get_attr(const std::string &key) { return attrs.at(key); };

  // void add_attr(std::string key, SamplesIndex value);
  // void add_attr(std::string key, SamplesReal value);

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

  /**
   * \todo debug me!
   */
  void init_icososphere(size_t num_refinements = 0) {

    MeshSamples ms = build_icososphere_half_edge_samples(num_refinements);

    from_mesh_samples(ms);
  }

  void save_ply(const std::string &filepath, const bool use_binary = true,
                const std::string &ply_property_convention = "MathUtils") {

    mathutils::mesh::io::save_mesh_samples(to_mesh_samples(), filepath,
                                           use_binary, ply_property_convention);
  }

  /**
   * @brief Compute V_cycle_E and V_cycle_F from HalfEdgeTopology.
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

  /**
   * \todo finish me!
   */
  bool collapse_edge(Index e);
};
/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils
