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
};

class SimplicialComplex2 {
public:
};

/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils
