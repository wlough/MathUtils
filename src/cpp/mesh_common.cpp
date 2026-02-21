/**
 * @file mesh_common.cpp
 *
 * @brief Common mesh data types and utilities.
 */

#include "mathutils/mesh/mesh_common.hpp"

namespace mathutils {
namespace mesh {

const SamplesVariant *get_variant_from_mesh_samples(const MeshSamples &ms,
                                                    std::string_view key) {
  auto it = ms.find(key); // no std::string construction
  if (it == ms.end())
    return nullptr;
  return &it->second;
}

bool erase_variant_from_mesh_samples(MeshSamples &ms, std::string_view key) {
  return ms.erase(std::string(key)) != 0;
}

} // namespace mesh
} // namespace mathutils
