#pragma once

/**
 * @file simplicial_complex2.hpp
 * @brief Two-dimensional simplicial complex.
 */
// #include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/simple_generator.hpp"
// #include "mathutils/mesh/mesh_plyio.hpp"
// #include "mathutils/simple_generator.hpp"
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
  using Generatori = mathutils::SimpleGenerator<Index>;

public:
  SamplesIndex V_cycle_E;
  SamplesIndex V_cycle_F;

  SamplesIndex f_incident_V;
  SamplesIndex F_incident_E;
  SamplesIndex E_incident_F;

  SimplicialTopology2() = default;
  SimplicialTopology2(size_t Ne, size_t Nf)
      : V_cycle_E(SamplesIndex(Ne, 2)), V_cycle_F(SamplesIndex(Nf, 3)) {}

  std::span<Index> V_cycle_e(Index e) { return V_cycle_E.row_span(e); }
  std::span<Index> V_cycle_f(Index f) { return V_cycle_F.row_span(f); }

  void from_mesh_samples(MeshSamples &ms);
  MeshSamples to_mesh_samples() const;
  std::map<std::string_view, SamplesIndex> to_topo_samples() const;

  Generatori generate_E_incident_f(Index f) const {
    co_yield E_incident_F(f, 0);
    co_yield E_incident_F(f, 1);
    co_yield E_incident_F(f, 2);
  }

  Generatori generate_V_cycle_f(Index f) const {
    co_yield V_cycle_F(f, 0);
    co_yield V_cycle_F(f, 1);
    co_yield V_cycle_F(f, 2);
  }

  Generatori generate_F_incident_v(Index v) const {

    Index f_start = f_incident_V[v];
    Index f = f_start;
    do {
      co_yield f;
      Index e0 = E_incident_F(f, 0);
      Index e1 = E_incident_F(f, 1);
    } while (f != f_start);
  }
  // Generatori generate_E_incident_v(Index v) const {
  //
  //   for (auto h : generate_H_rotcw_h(h_out_V[v])) {
  //     co_yield e_undirected_H[h];
  //   }
  // }
  // Generatori generate_V_adjacent_v(Index v) const {
  //
  //   // for (auto h : generate_H_rotcw_h(h_out_V[v])) {
  //   //   co_yield v_head_h(h);
  //   // }
  //   Index h_start = h_out_V[v];
  //   Index h = h_start;
  //   do {
  //     co_yield v_head_h(h);
  //     h = h_rotcw_h(h);
  //   } while (h != h_start);
  // }
};

class SimplicialComplex2 {
public:
};

/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils
