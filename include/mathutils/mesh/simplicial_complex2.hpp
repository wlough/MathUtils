#pragma once

/**
 * @file simplicial_complex2.hpp
 * @brief Two-dimensional simplicial complex.
 */
#include "mathutils/mesh/mesh_common.hpp"
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
public:
  SamplesIndex V_cycle_E_;
  SamplesIndex V_cycle_F_;

  SamplesIndex f_incident_V_;
  SamplesIndex F_incident_E_;
  SamplesIndex E_incident_F_;

  SimplicialTopology2() = default;
  SimplicialTopology2(size_t Ne, size_t Nf)
      : V_cycle_E_(SamplesIndex(Ne, 2)), V_cycle_F_(SamplesIndex(Nf, 3)) {}

  SamplesIndex &V_cycle_E() { return V_cycle_E_; }
  SamplesIndex &V_cycle_F() { return V_cycle_F_; }

  std::span<Index> V_cycle_e(Index e) { return V_cycle_E_.row_span(e); }
  std::span<Index> V_cycle_f(Index f) { return V_cycle_F_.row_span(f); }
};

/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils
