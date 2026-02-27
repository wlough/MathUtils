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

MeshSamples build_icosohedron_simplicial_samples();

MeshSamples build_icosohedron_half_edge_samples();

void refine_vertex_face_samples(MeshSamples &ms);

/**
 * @brief Divides face by adding a new vertex at the barycenter of the face
 *
 * @param f
 * @details
 * ```
 *                 v2                                    v2
 *               /   \                                 /   \
 *              /     \                               /     \
 *             /       \                             /       \
 *            /         \                           /    f2   \
 *           /           \                         /e7       e6\
 *          /             \                       /             \
 *         /               \                     /      e8       \
 *        /e2             e1\                   v6---------------v5
 *       /        f0         \                 /   \           /   \
 *      /                     \               /     \    f3   /     \
 *     /                       \             /   f0  \       /   f1  \
 *    /                         \           /e2     e1\     /e5     e4\
 *   /                           \         /           \   /           \
 *  /             e0              \ ----> /     e0      \ /     e3      \
 * v0 ----------------------------v1     v0 -------------v4-------------v1
 * ```
 */
void refine_simplicial_samples(MeshSamples &ms);

MeshSamples build_icososphere_simplicial_samples(size_t num_refinements);

} // namespace mesh
} // namespace mathutils
