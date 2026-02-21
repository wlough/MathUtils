#pragma once

/**
 * @file mesh_common_data_types.hpp
 */

// #include "mathutils/io/tinyply.h"
#include "mathutils/matrix.hpp"
// #include "mathutils/simple_generator.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cstdint>
#include <cstring>
// #include <iostream> // ***
// #include <limits>
#include <map>
// #include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/////////////////////////////////////
/////////////////////////////////////
// Mesh data types //////////////////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {
/** @addtogroup Mesh
 *  @{
 */
using Index = std::int64_t;
using Real = double;
using Color = std::uint8_t;

static_assert(std::is_integral<Index>::value, "Index must be an integral type");
static_assert(std::is_floating_point<Real>::value,
              "Real must be a floating point type");
static_assert(std::is_unsigned<Color>::value,
              "Color must be an unsigned integral type");
using SamplesIndex = Matrix<Index>;
using SamplesReal = Matrix<Real>;
using SamplesColor = Matrix<Color>;
using SamplesVariant = std::variant<SamplesReal, SamplesIndex, SamplesColor>;

using MeshSamples = std::map<std::string, SamplesVariant, std::less<>>;
// using MeshSamples =
//     std::map<std::string, SamplesVariant, TransparentStringLess>;
// using MeshSamples = std::map<std::string, SamplesVariant>;

static constexpr Index InvalidIndex = std::numeric_limits<Index>::max();

// const SamplesVariant *get_variant_from_mesh_samples(const MeshSamples &ms,
//                                                     std::string_view key);

const SamplesVariant *get_variant_from_mesh_samples(const MeshSamples &ms,
                                                    std::string_view key);

bool erase_variant_from_mesh_samples(MeshSamples &ms, std::string_view key);

template <typename T>
bool pop_variant_to_mat_from_mesh_samples(std::string_view key,
                                          Matrix<T> &out_mat, MeshSamples &ms) {
  if (const SamplesVariant *v = get_variant_from_mesh_samples(ms, key)) {
    assign_matrix_from_variant(*v, out_mat);
  }
  return erase_variant_from_mesh_samples(ms, key);
}

/**
@} // addtogroup Mesh
*/
} // namespace mesh
} // namespace mathutils

namespace mathutils {
namespace mesh {
////////////////////////////////
// Old types ///////////////////
////////////////////////////////
template <typename DataType, std::size_t dim>
using SamplesTypeDimTemplate =
    Eigen::Matrix<DataType, Eigen::Dynamic, dim,
                  (dim == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
template <typename DataType>
using RaggedSamplesTypeTemplate = std::vector<std::vector<DataType>>;

using Samplesi = SamplesTypeDimTemplate<int, 1>;
using Samples2i = SamplesTypeDimTemplate<int, 2>;
using Samples3i = SamplesTypeDimTemplate<int, 3>;

using Samplesd = Eigen::VectorXd;
using Samples2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Samples4d = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

template <typename IntType>
using MeshSamplesVariantTemplate = std::variant<
    SamplesTypeDimTemplate<IntType, 1>, SamplesTypeDimTemplate<IntType, 2>,
    SamplesTypeDimTemplate<IntType, 3>, SamplesTypeDimTemplate<IntType, 4>,
    SamplesTypeDimTemplate<IntType, 5>, SamplesTypeDimTemplate<IntType, 6>,
    SamplesTypeDimTemplate<IntType, 7>, SamplesTypeDimTemplate<IntType, 8>,
    Eigen::VectorXd, Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
    Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>,
    std::vector<std::vector<IntType>>>;

template <typename IntType>
using MeshSamplesTemplate =
    std::map<std::string, MeshSamplesVariantTemplate<IntType>>;

using MeshSamples32 = MeshSamplesTemplate<std::int32_t>;

} // namespace mesh
} // namespace mathutils
