/**
 * @file mesh_common.cpp
 *
 * @brief Common mesh data types and utilities.
 */

#include "mathutils/mesh/mesh_common.hpp"
namespace mathutils {
namespace mesh {

// template <typename Mat64>
// static auto checked_cast_eigen_int_to_i32(const Mat64 &v) {
//   static_assert(std::is_same_v<typename Mat64::Scalar, int>);
//   if (v.size() > 0) {
//     const auto minv = v.array().minCoeff();
//     const auto maxv = v.array().maxCoeff();
//     if (minv < std::numeric_limits<std::int32_t>::min() ||
//         maxv > std::numeric_limits<std::int32_t>::max()) {
//       throw std::runtime_error("int -> int32 cast would overflow");
//     }
//   }
//   return v.template cast<std::int32_t>().eval();
// }

// static std::vector<std::int32_t>
// checked_cast_vec_i64_to_i32(const std::vector<std::int64_t> &in) {
//   std::vector<std::int32_t> out;
//   out.reserve(in.size());
//   for (std::int64_t x : in) {
//     if (x < std::numeric_limits<std::int32_t>::min() ||
//         x > std::numeric_limits<std::int32_t>::max()) {
//       throw std::runtime_error("int64 -> int32 cast would overflow");
//     }
//     out.push_back(static_cast<std::int32_t>(x));
//   }
//   return out;
// }

MeshSamples32
convert_mesh_samples_mixed_to_32(const MeshSamples &mesh_samples) {
  MeshSamples32 mesh_samples_32;

  for (const auto &[key, value] : mesh_samples) {
    std::visit(
        [&](const auto &v) {
          using T = std::decay_t<decltype(v)>;

          if constexpr (std::is_same_v<T, Samplesi64> ||
                        std::is_same_v<T, Samples2i64> ||
                        std::is_same_v<T, Samples3i64> ||
                        std::is_same_v<T, Samples4i64>) {
            mesh_samples_32[key] = checked_cast_eigen_i64_to_i32(v);

          } else if constexpr (std::is_same_v<T, RaggedSamplesi64>) {
            RaggedSamplesi32 ragged_32;
            ragged_32.reserve(v.size());
            for (const auto &vec64 : v) {
              ragged_32.emplace_back(checked_cast_vec_i64_to_i32(vec64));
            }
            mesh_samples_32[key] = std::move(ragged_32);

          } else {
            // Samples*i32, Samples*d, etc.
            mesh_samples_32[key] = v;
          }
        },
        value);
  }

  return mesh_samples_32;
}

} // namespace mesh
} // namespace mathutils