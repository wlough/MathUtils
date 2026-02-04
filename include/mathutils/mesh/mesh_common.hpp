#pragma once

/**
 * @file mesh_common_data_types.hpp
 */

#include "mathutils/io/tinyply.h"
#include "mathutils/matrix.hpp"
#include "mathutils/simple_generator.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream> // ***
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

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

// using Samplesi32 = SamplesTypeDimTemplate<std::int32_t, 1>;
// using Samples2i32 = SamplesTypeDimTemplate<std::int32_t, 2>;
// using Samples3i32 = SamplesTypeDimTemplate<std::int32_t, 3>;
// using Samples4i32 = SamplesTypeDimTemplate<std::int32_t, 4>;
// using Samples5i32 = SamplesTypeDimTemplate<std::int32_t, 5>;
// using Samples6i32 = SamplesTypeDimTemplate<std::int32_t, 6>;
// using Samples7i32 = SamplesTypeDimTemplate<std::int32_t, 7>;
// using Samples8i32 = SamplesTypeDimTemplate<std::int32_t, 8>;

// using Samplesi64 = SamplesTypeDimTemplate<std::int64_t, 1>;
// using Samples2i64 = SamplesTypeDimTemplate<std::int64_t, 2>;
// using Samples3i64 = SamplesTypeDimTemplate<std::int64_t, 3>;
// using Samples4i64 = SamplesTypeDimTemplate<std::int64_t, 4>;
// using Samples5i64 = SamplesTypeDimTemplate<std::int64_t, 5>;
// using Samples6i64 = SamplesTypeDimTemplate<std::int64_t, 6>;
// using Samples7i64 = SamplesTypeDimTemplate<std::int64_t, 7>;
// using Samples8i64 = SamplesTypeDimTemplate<std::int64_t, 8>;

using Samplesi = SamplesTypeDimTemplate<int, 1>;
using Samples2i = SamplesTypeDimTemplate<int, 2>;
using Samples3i = SamplesTypeDimTemplate<int, 3>;
// using Samples4i = SamplesTypeDimTemplate<int, 4>;
// using Samples5i = SamplesTypeDimTemplate<int, 5>;
// using Samples6i = SamplesTypeDimTemplate<int, 6>;
// using Samples7i = SamplesTypeDimTemplate<int, 7>;
// using Samples8i = SamplesTypeDimTemplate<int, 8>;

// using RaggedSamplesi32 = RaggedSamplesTypeTemplate<std::int32_t>;
// using RaggedSamplesi64 = RaggedSamplesTypeTemplate<std::int64_t>;
// using RaggedSamplesi = RaggedSamplesTypeTemplate<int>;
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
    Samplesd, Samples2d, Samples3d, Samples4d,
    std::vector<std::vector<IntType>>>;

template <typename IntType>
using MeshSamplesTemplate =
    std::map<std::string, MeshSamplesVariantTemplate<IntType>>;

// using VarDict = std::map<std::string, std::variant<int, double, float, bool>>

// using MeshSamples = MeshSamplesTemplate<int>;
// using MeshSamples =
//     std::map<std::string,
//              std::variant<Samplesi, Samples2i, Samples3i, Samples4i,
//              Samples5i,
//                           Samples6i, Samples7i, Samples8i, Samplesd,
//                           Samples2d, Samples3d, Samples4d,
//                           std::vector<std::vector<int>>>>;
using MeshSamples32 = MeshSamplesTemplate<std::int32_t>;
// using MeshSamples64 = MeshSamplesTemplate<std::int64_t>;

// using MeshSamplesMixedVariant = std::variant<
//     Samplesi32, Samples2i32, Samples3i32, Samples4i32, Samples5i32,
//     Samples6i32, Samples7i32, Samples8i32, Samplesi64, Samples2i64,
//     Samples3i64, Samples4i64, Samples5i64, Samples6i64, Samples7i64,
//     Samples8i64, Samplesi, Samples2i, Samples3i, Samples4i, Samples5i,
//     Samples6i, Samples7i, Samples8i, Samplesd, Samples2d, Samples3d,
//     Samples4d, RaggedSamplesi32, RaggedSamplesi64>;

// using MeshSamplesMixed = std::map<std::string, MeshSamplesMixedVariant>;

} // namespace mesh
} // namespace mathutils

/////////////////////////////////////
/////////////////////////////////////
// Mesh data types //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {

using Index = std::int64_t;
using Real = double;
using Color = std::uint8_t;
// static_assert(std::is_unsigned<Index>::value,
//               "Index must be an unsigned integral type");
static_assert(std::is_integral<Index>::value, "Index must be an integral type");
static_assert(std::is_floating_point<Real>::value,
              "Real must be a floating point type");
static_assert(std::is_unsigned<Color>::value,
              "Color must be an unsigned integral type");
using SamplesIndex = Matrix<Index>;
using SamplesField = Matrix<Real>;
using SamplesRGBA = Matrix<Color>;
using SamplesVariant = std::variant<SamplesField, SamplesIndex, SamplesRGBA>;
using MeshSamples = std::map<std::string, SamplesVariant>;

} // namespace mesh
} // namespace mathutils
