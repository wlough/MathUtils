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
static_assert(std::is_floating_point<Real>::value,
              "Real must be a floating point type");
static_assert(std::is_unsigned<Color>::value,
              "Color must be an unsigned integral type");
using SamplesIndex = Matrix<Index>;
using SamplesField = Matrix<Real>;
using SamplesRGBA = Matrix<Color>;
using SamplesVariant = std::variant<SamplesIndex, SamplesField, SamplesRGBA>;
using MeshSamples = std::map<std::string, SamplesVariant>;

using PlyIndex = std::int32_t;
using PlyReal = double;
using PlyColor = std::uint8_t;
static tinyply::Type tinyplyIndex = tinyply::Type::INT32;
static tinyply::Type tinyplyReal = tinyply::Type::FLOAT64;
static tinyply::Type tinyplyColor = tinyply::Type::UINT8;
using PlySamplesIndex = Matrix<PlyIndex>;
using PlySamplesField = Matrix<PlyReal>;
using PlySamplesRGBA = Matrix<PlyColor>;
using PlySamplesVariant =
    std::variant<PlySamplesIndex, PlySamplesField, PlySamplesRGBA>;
using PlyMeshSamples = std::map<std::string, PlySamplesVariant>;

enum class SampleType : uint8_t { INDEX, FIELD, COLOR, INVALID };

static std::map<SampleType, tinyply::Type> PlyTypeFromSampleType{
    {SampleType::INDEX, tinyply::Type::UINT32},
    {SampleType::FIELD, tinyply::Type::FLOAT64},
    {SampleType::COLOR, tinyply::Type::UINT8},
    {SampleType::INVALID, tinyply::Type::INVALID}};
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

enum IndexState : Index {
  MESH_ID_VERTEX = 0,
  MESH_ID_EDGE = 1,
  MESH_ID_FACE = 2,
  MESH_ID_CELL = 3,
  MESH_ID_BOUNDARY = 4,
  MESH_ID_HALF_EDGE = 5,
  MESH_ID_DART = 6,
  MESH_ID_SIMPLEX = 7,
  MESH_ID_UNASSIGNED = std::numeric_limits<Index>::max() - 1,
  MESH_ID_INVALID = std::numeric_limits<Index>::max(),
};

class MeshObjectIndexed {
public:
  std::string name;
  virtual ~MeshObjectIndexed() = default;
  // virtual to_ply_spec();
  std::vector<std::string> id_data_keys;
  std::vector<SamplesIndex> id_data;
  std::vector<std::string> field_data_keys;
  std::vector<SamplesField> field_data;
  std::vector<std::string> rgba_data_keys;
  std::vector<SamplesRGBA> rgba_data;

  void add_id_data(const std::string &key, const SamplesIndex &dat) {
    auto it = std::find(id_data_keys.begin(), id_data_keys.end(), key);
    if (it != id_data_keys.end()) {
      throw std::invalid_argument("id_data key " + key + " already exists");
    }
    id_data_keys.push_back(key);
    id_data.push_back(dat);
  }
  void add_field_data(const std::string &key, const SamplesField &dat) {
    auto it = std::find(field_data_keys.begin(), field_data_keys.end(), key);
    if (it != field_data_keys.end()) {
      throw std::invalid_argument("field_data key " + key + " already exists");
    }
    field_data_keys.push_back(key);
    field_data.push_back(dat);
  }
  void add_rgba_data(const std::string &key, const SamplesRGBA &dat) {
    auto it = std::find(rgba_data_keys.begin(), rgba_data_keys.end(), key);
    if (it != rgba_data_keys.end()) {
      throw std::invalid_argument("rgba_data key " + key + " already exists");
    }
    rgba_data_keys.push_back(key);
    rgba_data.push_back(dat);
  }
};

class VertexSetIndexed : public MeshObjectIndexed {
  using ID = Index;
  using DataID = SamplesIndex;
  using DataField = SamplesField;
  using DataRGBA = SamplesRGBA;

public:
  VertexSetIndexed() {
    name = "vertex";
    id_data_keys = {"h_out_V", "d_through_V"};
    field_data_keys = {"xyz_coord_V"};
    rgba_data_keys = {"rgba_V"};
  };

  DataID &get_h_out_V() { return id_data[0]; }
  DataID &get_d_through_V() { return id_data[1]; }
  DataField &get_xyz_coord_V() { return field_data[0]; }
  DataRGBA &get_rgba_V() { return rgba_data[0]; }

  ID h_out_v(ID v) const { return id_data[0][v]; }
  ID d_through_v(ID v) const { return id_data[1][v]; }
};

class EdgeSetIndexed : public MeshObjectIndexed {

public:
  EdgeSetIndexed() {
    name = "edge";
    id_data_keys = {"V_cycle_E", "h_directed_E", "d_through_E"};
  };
};

class FaceSetIndexed : public MeshObjectIndexed {
public:
};

class CellSetIndexed : public MeshObjectIndexed {
public:
};

class BoundarySetIndexed : public MeshObjectIndexed {
public:
};

class HalfEdgeSetIndexed : public MeshObjectIndexed {
public:
};

class DartSetIndexed : public MeshObjectIndexed {
public:
};

// template <typename ID, typename Int> struct HalfEdgeMapTemplate {
//   using Generatori = mathutils::GeneratorTemplate<ID>;
//   /////////////////////////
//   // Core data structure //
//   /////////////////////////
//   /**
//    * @brief Matrix vefnt_H[h] = [v_origin, e_undirected, f_left, h_next,
//    h_twin]
//    */
//   MeshDataStatic<ID, 1> h_out_V, h_directed_E, h_right_F, h_negative_B,
//       v_origin_H, e_undirected_H, f_left_H, h_next_H, h_twin_H;
//   ///////////////////////////////////////////
//   // Fundamental accessors and properties ///
//   ///////////////////////////////////////////
//   ID h_out_v(ID v) const { return h_out_V(v); }
//   ID h_directed_e(ID e) const { return h_directed_E(e); }
//   ID h_right_f(ID f) const { return h_right_F(f); }
//   ID h_negative_b(ID b) const { return h_negative_B(b); }

//   ID v_origin_h(ID h) const { return vefnt_H(h, 0); }
//   ID e_undirected_h(ID h) const { return vefnt_H(h, 1); }
//   ID f_left_h(ID h) const { return vefnt_H(h, 2); }

//   ID h_next_h(ID h) const { return vefnt_H(h, 3); }
//   ID h_twin_h(ID h) const { return vefnt_H(h, 4); }

//   ID v_head_h(ID h) const { return vefnt_H(vefnt_H(h, 4), 0); }
//   ID h_rotcw_h(ID h) const { return vefnt_H(vefnt_H(h, 4), 3); }
//   ID num_vertices() const { return h_out_V.rows(); }
//   ID num_edges() const { return vefnt_H.rows() / 2; }
//   ID num_faces() const { return h_right_F.rows(); }
//   ID num_boundaries() const { return h_negative_B.rows(); }
//   ID num_half_edges() const { return vefnt_H.rows(); }
//   Int euler_characteristic() const {
//     return static_cast<Int>(num_vertices()) - static_cast<Int>(num_edges()) +
//            static_cast<Int>(num_faces());
//   }
//   bool some_negative_boundary_contains_h(ID h) const {
//     ID f = f_left_h(h);
//     return (f >= num_faces());
//   }
//   ID b_ghost_f(ID f) const { return f - num_faces(); }

//   Generatori generate_H_next_h(ID h) const {
//     ID h_start = h;
//     do {
//       co_yield h;
//       h = h_next_h(h);
//     } while (h != h_start);
//   }

//   Generatori generate_H_rotcw_h(ID h) const {
//     ID h_start = h;
//     do {
//       co_yield h;
//       h = h_rotcw_h(h);
//     } while (h != h_start);
//   }

//   MeshSamplesTemplate<ID> to_mesh_samples() const {
//     MeshSamplesTemplate<ID> ms;
//     ms["h_out_V"] = h_out_V;
//     ms["h_directed_E"] = h_directed_E;
//     ms["h_right_F"] = h_right_F;
//     ms["h_negative_B"] = h_negative_B;
//     ms["v_origin_H"] = vefnt_H.col(0);
//     ms["e_undirected_H"] = vefnt_H.col(1);
//     ms["f_left_H"] = vefnt_H.col(2);
//     ms["h_next_H"] = vefnt_H.col(3);
//     ms["h_twin_H"] = vefnt_H.col(4);
//     return ms;
//   }
// };

// /**
//  * @brief Check if int -> int32 cast would overflow, and perform the cast.
//  *
//  * @param v
//  * @return * template <typename Mat64>
//  */
// template <typename Mat64>
// static auto checked_cast_eigen_int_to_i32(const Mat64 &v);
// /**
//  * @brief Check if int-> int32 cast would overflow, and perform the cast.
//  *
//  * @param in
//  * @return  std::vector<std::int32_t>
//  */
// static std::vector<std::int32_t>
// checked_cast_vec_int_to_i32(const std::vector<int> &in);

/**
 * @brief Convert MeshSamples to MeshSamples32 by casting integer types.
 */
// MeshSamples32 convert_mesh_samples_to_32(const MeshSamples &mesh_samples);

// template <typename IntType> struct CellComplexTemplate {
//   RaggedSamplesTypeTemplate<IntType> V_cycle_E;
//   RaggedSamplesTypeTemplate<IntType> V_cycle_F;
//   RaggedSamplesTypeTemplate<IntType> V_cycle_C;
// };

// template <typename IntType> struct SimplicialComplexTemplate {
//   SamplesiTemplate<IntType> V_cycle_E;
//   Samples2iTemplate<IntType> V_cycle_F;
//   Samples3iTemplate<IntType> V_cycle_C;
// };

// struct HalfFaceData {
//   Eigen::VectorXi h_out_V;
//   Eigen::VectorXi h_directed_E;
//   Eigen::VectorXi h_right_F;
//   Eigen::VectorXi h_above_C;
//   Eigen::VectorXi h_negative_B;

//   Eigen::VectorXi v_origin_H;
//   Eigen::VectorXi e_undirected_H;
//   Eigen::VectorXi f_left_H;
//   Eigen::VectorXi c_below_H;

//   Eigen::VectorXi h_next_H;
//   Eigen::VectorXi h_twin_H;
//   Eigen::VectorXi h_flip_H;
// };

// struct CombinatorialMap2Data {
//   Eigen::VectorXi d_through_S0;
//   Eigen::VectorXi d_through_S1;
//   Eigen::VectorXi d_through_S2;

//   Eigen::VectorXi s0_in_D;
//   Eigen::VectorXi s1_in_D;
//   Eigen::VectorXi s2_in_D;

//   Eigen::VectorXi d_cmap0_D;
//   Eigen::VectorXi d_cmap1_D;
//   Eigen::VectorXi d_cmap2_D;
// };

// template <typename IntType> struct HalfEdgeMapTemplate {
//   using Index = Eigen::Index;
//   using Samples5i = Eigen::Matrix<IntType, Eigen::Dynamic, 5,
//   Eigen::RowMajor>; using Samplesi = Eigen::Matrix<IntType, Eigen::Dynamic,
//   1, Eigen::RowMajor>; using Generatori =
//   mathutils::GeneratorTemplate<IntType>;

//   /////////////////////////
//   // Core data structure //
//   /////////////////////////
//   /**
//    * @brief Matrix vefnt_H[h] = [v_origin, e_undirected, f_left, h_next,
//    h_twin]
//    */
//   Samples5i vefnt_H;
//   Samplesi h_out_V, h_directed_E, h_right_F, h_negative_B;

//   ///////////////////////////////////////////
//   // Fundamental accessors and properties ///
//   ///////////////////////////////////////////
//   IntType h_out_v(Index v) const { return h_out_V(v); }
//   IntType h_directed_e(Index e) const { return h_directed_E(e); }
//   IntType h_right_f(Index f) const { return h_right_F(f); }
//   IntType h_negative_b(Index b) const { return h_negative_B(b); }

//   IntType v_origin_h(Index h) const { return vefnt_H(h, 0); }
//   IntType e_undirected_h(Index h) const { return vefnt_H(h, 1); }
//   IntType f_left_h(Index h) const { return vefnt_H(h, 2); }

//   IntType h_next_h(IntType h) const { return vefnt_H(h, 3); }
//   IntType h_twin_h(Index h) const { return vefnt_H(h, 4); }

//   IntType v_head_h(Index h) const { return vefnt_H(vefnt_H(h, 4), 0); }
//   IntType h_rotcw_h(Index h) const { return vefnt_H(vefnt_H(h, 4), 3); }

//   Index num_vertices() const { return h_out_V.size(); }
//   Index num_edges() const { return vefnt_H.rows() / 2; }
//   Index num_faces() const { return h_right_F.size(); }
//   Index num_boundaries() const { return h_negative_B.size(); }
//   Index num_half_edges() const { return vefnt_H.rows(); }
//   IntType euler_characteristic() const {
//     return num_vertices() - num_edges() + num_faces();
//   }

//   // bool some_negative_boundary_contains_h(Index h) const {
//   //   // return f_left_h(h) < 0;
//   //   IntType f = f_left_h(h);
//   //   return (f < 0) || (f >= num_faces());
//   // }
//   bool some_negative_boundary_contains_h(Index h) const {
//     return (f_left_h(h) >= num_faces());
//   }
//   IntType b_ghost_f(IntType f) const { return -f - 1; }

//   Generatori generate_H_next_h(Index h) const {
//     IntType h_start = h;
//     do {
//       co_yield h;
//       h = h_next_h(h);
//     } while (h != h_start);
//   }

//   Generatori generate_H_rotcw_h(Index h) const {
//     int h_start = h;
//     do {
//       co_yield h;
//       h = h_rotcw_h(h);
//     } while (h != h_start);
//   }

//   MeshSamplesTemplate<IntType> to_mesh_samples() const {
//     MeshSamplesTemplate<IntType> ms;
//     ms["h_out_V"] = h_out_V;
//     ms["h_directed_E"] = h_directed_E;
//     ms["h_right_F"] = h_right_F;
//     ms["h_negative_B"] = h_negative_B;
//     ms["v_origin_H"] = vefnt_H.col(0);
//     ms["e_undirected_H"] = vefnt_H.col(1);
//     ms["f_left_H"] = vefnt_H.col(2);
//     ms["h_next_H"] = vefnt_H.col(3);
//     ms["h_twin_H"] = vefnt_H.col(4);
//     return ms;
//   }

//   // SimplicialComplexTemplate<IntType> to_simplicial_complex() const {
//   //   SimplicialComplexTemplate<IntType> sc;
//   //   sc.V_cycle_E.resize(h_directed_E.size(), 2);
//   //   for (Index e = 0; e < h_directed_E.size(); ++e) {
//   //     IntType h = h_directed_E(e);
//   //     sc.V_cycle_E(e, 0) = v_origin_h(h);
//   //     sc.V_cycle_E(e, 1) = v_origin_h(h_twin_h(h));
//   //   }
//   //   sc.V_cycle_F.resize(h_right_F.size(), 3);
//   //   for (Index f = 0; f < h_right_F.size(); ++f) {
//   //     IntType h = h_right_F(f);
//   //     for (Index i = 0; i < 3; ++i) {
//   //       sc.V_cycle_F(f, i) = v_origin_h(h);
//   //       h = h_next_h(h);
//   //     }
//   //   }
//   //   return sc;
//   // }
// };

// using HalfEdgeMap = HalfEdgeMapTemplate<int>;
// using HalfEdgeMap32 = HalfEdgeMapTemplate<std::int32_t>;
// using HalfEdgeMap64 = HalfEdgeMapTemplate<std::int64_t>;

// /**
//  * @brief Template struct for a vertex set where template parameter is
//  dimension
//  * of the space in which the vertices live.
//  *
//  */
// template <int Dim> struct VertexSetTemplate {
//   using SamplesNd = Eigen::Matrix<double, Eigen::Dynamic, Dim,
//   Eigen::RowMajor>; using RowCoordsNd = Eigen::Matrix<double, 1, Dim,
//   Eigen::RowMajor>; using Index = Eigen::Index;

//   SamplesNd coord_V;

//   static constexpr int dimension() { return Dim; }
//   Index num_vertices() const { return coord_V.rows(); }

//   RowCoordsNd coord_v(Index v) const { return coord_V.row(v).eval(); }
// };

// using VertexSetDim2 = VertexSetTemplate<2>;
// using VertexSetDim3 = VertexSetTemplate<3>;

} // namespace mesh
} // namespace mathutils
