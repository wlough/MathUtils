#pragma once

/**
 * @file mesh_builder_funs.hpp
 * @brief Functions for initializing/building meshes
 */

#include "mathutils/mesh/half_edge_mesh.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/simplicial_complex2.hpp"

namespace mathutils {
namespace mesh {
MeshSamples build_icosohedron();

void refine_vertex_face_samples(MeshSamples &ms);

MeshSamples build_icososphere(size_t num_refinements);
} // namespace mesh
} // namespace mathutils
