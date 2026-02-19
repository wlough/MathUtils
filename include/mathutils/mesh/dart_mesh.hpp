#pragma once

/**
 * @file half_edge_mesh.hpp
 * @brief Simple half-edge mesh class
 */
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/simple_generator.hpp"
#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
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

struct Simplex {
  Index id_;
  SamplesReal pts_;
  Simplex() = default;
  Simplex(const Index id, const SamplesReal pts) : id_(id), pts_(pts) {}
};

class DartMesh {
public:
  size_t dimension_{2};
  SamplesReal S0_;
  SamplesIndex S1_;
  SamplesIndex S2_;
  SamplesIndex S3_;
};

} // namespace mesh
} // namespace mathutils
