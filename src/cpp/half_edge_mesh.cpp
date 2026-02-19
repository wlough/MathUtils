#include "mathutils/mesh/half_edge_mesh.hpp"

namespace mathutils {
namespace mesh {
//
//
void HalfPlexMesh::from_mesh_samples(const MeshSamples &ms) {
  auto get = [&](const char *key) -> const SamplesVariant * {
    auto it = ms.find(key);
    if (it == ms.end())
      return nullptr;
    return &it->second;
  };

  if (auto v = get("h_out_V")) {
    assign_matrix_from_variant(*v, "h_out_V", h_out_V_);
  }
  if (auto v = get("h_directed_E")) {
    assign_matrix_from_variant(*v, "h_directed_E", h_directed_E_);
  }
  if (auto v = get("h_right_F")) {
    assign_matrix_from_variant(*v, "h_right_F", h_right_F_);
  }
  if (auto v = get("h_above_C")) {
    assign_matrix_from_variant(*v, "h_above_C", h_above_C_);
  }
  if (auto v = get("h_negative_B")) {
    assign_matrix_from_variant(*v, "h_negative_B", h_negative_B_);
  }

  if (auto v = get("v_origin_H")) {
    assign_matrix_from_variant(*v, "v_origin_H", v_origin_H_);
  }
  if (auto v = get("e_undirected_H")) {
    assign_matrix_from_variant(*v, "e_undirected_H", e_undirected_H_);
  }
  if (auto v = get("f_left_H")) {
    assign_matrix_from_variant(*v, "f_left_H", f_left_H_);
  }
  if (auto v = get("c_below_H")) {
    assign_matrix_from_variant(*v, "c_below_H", c_below_H_);
  }

  if (auto v = get("h_next_H")) {
    assign_matrix_from_variant(*v, "h_next_H", h_next_H_);
  }
  if (auto v = get("h_twin_H")) {
    assign_matrix_from_variant(*v, "h_twin_H", h_twin_H_);
  }
  if (auto v = get("h_flip_H")) {
    assign_matrix_from_variant(*v, "h_flip_H", h_flip_H_);
  }

  if (auto v = get("X_ambient_V")) {
    assign_matrix_from_variant(*v, "X_ambient_V", X_ambient_V_);
  }
  if (auto v = get("V_cycle_E")) {
    assign_matrix_from_variant(*v, "V_cycle_E", V_cycle_E_);
  }
  if (auto v = get("V_cycle_F")) {
    assign_matrix_from_variant(*v, "V_cycle_F", V_cycle_F_);
  }
  if (auto v = get("V_cycle_C")) {
    assign_matrix_from_variant(*v, "V_cycle_C", V_cycle_C_);
  }
}

} // namespace mesh
} // namespace mathutils
