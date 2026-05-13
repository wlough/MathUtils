#pragma once

/**
 * @file mesh_builder_funs.hpp
 * @brief Functions for initializing/building meshes
 */

#include "mathutils/mesh/mesh_common.hpp"
#include <map>

namespace mathutils {
namespace mesh {

/**
 * @brief Find the index ht of H[ht]=[j, i] in H, where H[h]=[i, j]. Return
 * InvalidIndex if not found.
 *
 * @param H Half-edge samples (Nh, 2)
 * @param h Index of half-edge in H
 * @return Index of twin half-edge in H, or InvalidIndex if not found
 */
Index find_halfedge_index_of_twin(const SamplesIndex &H, const Index &h);

/**
 * @brief Convert triangle vertex cycles to edge vertex cycles.
 *
 * @param V_cycle_F triangle vertex cycles (Nh, 3)
 * @return SamplesIndex V_cycle_E edge vertex cycles (Nh, 2)
 */
SamplesIndex tri_cycles_to_edge_cycles(const SamplesIndex &V_cycle_F);

/**
 * @brief Convert triangle vertex cycles to half-edge samples.
 *
 * @param V_cycle_F (Nf, 3) triangle vertex cycles
 * @return std::map<std::string, Samplesi> Half-edge samples and related data
 */
std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples_alt(const SamplesIndex &V_cycle_F);

/**
 * @brief Convert triangle vertex cycles to half-edge samples.
 *
 * @param V_cycle_F (Nf, 3) triangle vertex cycles
 * @return std::map<std::string, SampleSimple half-edge mesh classsi> Half-edge
 * samples and related data
 */
std::map<std::string, SamplesIndex>
tri_cycles_to_half_edge_samples(const SamplesIndex &V_cycle_F);

std::map<std::string, SamplesIndex>
edge_tri_cycles_to_half_edge_samples(const SamplesIndex &V_cycle_E,
                                     const SamplesIndex &V_cycle_F);

std::map<std::string, SamplesIndex>
half_edge_samples_no_edge_data_to_edge_tri_cycles(
    const std::map<std::string, SamplesIndex> &he_samples);

std::map<std::string, SamplesIndex> half_edge_samples_to_edge_tri_cycles(
    const std::map<std::string, SamplesIndex> &he_samples);

std::map<std::string, SamplesIndex>
tri_cycles_to_dart_samples(const SamplesIndex &V_cycle_F);

std::map<std::string, SamplesIndex> half_edge_samples_to_dart_samples(
    const std::map<std::string, SamplesIndex> &ms);

std::map<std::string, SamplesIndex> half_edge_samples_to_simplicial_samples(
    const std::map<std::string, SamplesIndex> &he_samples);

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

MeshSamples build_icososphere_half_edge_samples(size_t num_refinements);

} // namespace mesh
} // namespace mathutils
