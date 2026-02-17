#pragma once

/**
 * @file mesh_plyio.hpp
 * @brief Mesh input/output.
 */

#include "mathutils/mesh/mesh_common.hpp"
#include <chrono> // std::chrono::high_resolution_clock and std::chrono::duration
#include <fstream>   // std::ifstream
#include <istream>   // std::istream
#include <map>       // std::map
#include <set>       // std::set
#include <stdexcept> // std::runtime_error
#include <streambuf> // std::streambuf
#include <string>    // std::string
#include <variant>   // std::variant
#include <vector>    // std::vector

/**
 * @defgroup MeshIO Mesh input/output
 * @brief Tools for reading and writing mesh data.
 * @details The `MeshLoader` class uses the tinyply library for reading and
 * writing .ply files.
 */

namespace mathutils {
namespace mesh {
namespace io {
// Data types for PLY I/O. Must be compatible with tinyply types.
using PlyIndex = std::int32_t;
using PlyReal = double;
using PlyColor = std::uint8_t;
using PlySamplesIndex = Matrix<PlyIndex>;
using PlySamplesField = Matrix<PlyReal>;
using PlySamplesRGBA = Matrix<PlyColor>;
using PlySamplesVariant =
    std::variant<PlySamplesField, PlySamplesIndex, PlySamplesRGBA>;
using PlyMeshSamples = std::map<std::string, PlySamplesVariant>;

enum class SampleType : uint8_t { INDEX, FIELD, COLOR, INVALID };

static std::map<SampleType, tinyply::Type> TinyplyTypeFromSampleType{
    {SampleType::INDEX, tinyply::Type::UINT32},
    {SampleType::FIELD, tinyply::Type::FLOAT64},
    {SampleType::COLOR, tinyply::Type::UINT8},
    {SampleType::INVALID, tinyply::Type::INVALID}};

/**
 * @brief Specification for a properties of an element in a PLY mesh file.
 *
 *
 */
struct MeshPlyPropertySpec {
  //////////////////
  // Constructors //
  //////////////////
  MeshPlyPropertySpec() = default;
  MeshPlyPropertySpec(const std::string element_key_,
                      const std::string samples_key_,
                      const std::vector<std::string> property_keys_,
                      const SampleType sample_type_, bool is_list_)
      : element_key(element_key_), samples_key(samples_key_),
        property_keys(property_keys_), sample_type(sample_type_),
        is_list(is_list_) {}
  MeshPlyPropertySpec(const std::string element_key_,
                      const std::string samples_key_,
                      const std::vector<std::string> property_keys_,
                      const SampleType sample_type_)
      : element_key(element_key_), samples_key(samples_key_),
        property_keys(property_keys_), sample_type(sample_type_),
        is_list(false) {}
  MeshPlyPropertySpec(const std::string element_key_,
                      const std::string samples_key_,
                      const std::vector<std::string> property_keys_,
                      const SampleType sample_type_, std::size_t list_count_)
      : element_key(element_key_), samples_key(samples_key_),
        property_keys(property_keys_), sample_type(sample_type_),
        is_list(list_count_ > 0), list_count(list_count_) {}

  ///////////////////
  // Keys and data //
  ///////////////////
  std::string element_key;
  std::string samples_key;
  std::vector<std::string> property_keys;
  SampleType sample_type{SampleType::INVALID};
  bool is_list{false};
  std::size_t list_count{0};

  /**
   * @brief Check for `Matrix`-valued property with key `samples_key` in
   * `mesh_samples` and add it to `mesh_file` if found.
   *
   * @param mesh_samples The mesh samples containing the data.
   * @param mesh_file The mesh file to which the property will be added.
   */
  void add_property_to_mesh_file(const PlyMeshSamples &mesh_samples,
                                 tinyply::PlyFile &mesh_file) const;

  std::shared_ptr<tinyply::PlyData>
  request_property_from_mesh_file(tinyply::PlyFile &mesh_file) const;
};

/**
 * @brief List of all known PLY property specifications.
 *
 */
static std::vector<MeshPlyPropertySpec> PlyPropertySpecs{
    ////////////
    // Vertex //
    ////////////
    MeshPlyPropertySpec("vertex", "xyz_coord_V", {"x", "y", "z"},
                        SampleType::FIELD, false),
    MeshPlyPropertySpec("vertex", "h_out_V", {"h_out"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("vertex", "d_through_V", {"d_through"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("vertex", "rgba_V", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////
    // Edge //
    //////////
    MeshPlyPropertySpec("edge", "V_cycle_E", {"V_cycle"}, SampleType::INDEX,
                        true),
    MeshPlyPropertySpec("edge", "h_directed_E", {"h_directed"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("edge", "d_through_E", {"d_through"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("edge", "rgba_E", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////
    // Face //
    //////////
    MeshPlyPropertySpec("face", "V_cycle_F", {"V_cycle"}, SampleType::INDEX,
                        true),
    MeshPlyPropertySpec("face", "h_right_F", {"h_right"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("face", "d_through_F", {"d_through"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("face", "rgba_F", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////
    // Cell //
    //////////
    MeshPlyPropertySpec("cell", "V_cycle_C", {"V_cycle"}, SampleType::INDEX,
                        true),
    MeshPlyPropertySpec("cell", "h_above_C", {"h_above"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("cell", "d_through_C", {"d_through"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("cell", "rgba_C", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////////
    // Boundary //
    //////////////
    MeshPlyPropertySpec("boundary", "h_negative_B", {"h_negative"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("boundary", "d_through_B", {"d_through"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("boundary", "rgba_B", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////////
    // HalfEdge //
    //////////////
    MeshPlyPropertySpec("half_edge", "v_origin_H", {"v_origin"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("half_edge", "e_undirected_H", {"e_undirected"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("half_edge", "f_left_H", {"f_left"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "c_below_H", {"c_below"},
                        SampleType::INDEX, false),
    MeshPlyPropertySpec("half_edge", "h_next_H", {"h_next"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "h_twin_H", {"h_twin"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "h_flip_H", {"h_flip"}, SampleType::INDEX,
                        false),
    //////////
    // Dart //
    //////////
    MeshPlyPropertySpec("dart", "s0_D", {"s0"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "s1_D", {"s1"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "s2_D", {"s2"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "s3_D", {"s3"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "d0_D", {"d0"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "d1_D", {"d1"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "d2_D", {"d2"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("dart", "d3_D", {"d3"}, SampleType::INDEX, false)};

static std::vector<MeshPlyPropertySpec> MeshBranePlyPropertySpecs{
    ////////////
    // Vertex //
    ////////////
    MeshPlyPropertySpec("vertex", "xyz_coord_V", {"x", "y", "z"},
                        SampleType::FIELD, false),
    MeshPlyPropertySpec("vertex", "h_out_V", {"h"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("vertex", "d_through_V", {"d"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("vertex", "rgba_V", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////
    // Edge //
    //////////
    MeshPlyPropertySpec("edge", "V_cycle_E", {"vertex_indices"},
                        SampleType::INDEX, true),
    MeshPlyPropertySpec("edge", "h_directed_E", {"h"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("edge", "d_through_E", {"d"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("edge", "rgba_E", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////
    // Face //
    //////////
    MeshPlyPropertySpec("face", "V_cycle_F", {"vertex_indices"},
                        SampleType::INDEX, true),
    MeshPlyPropertySpec("face", "h_right_F", {"h"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("face", "d_through_F", {"d"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("face", "rgba_F", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////
    // Cell //
    //////////
    MeshPlyPropertySpec("cell", "V_cycle_C", {"vertex_indices"},
                        SampleType::INDEX, true),
    MeshPlyPropertySpec("cell", "h_above_C", {"h"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("cell", "d_through_C", {"d"}, SampleType::INDEX, false),
    MeshPlyPropertySpec("cell", "rgba_C", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////////
    // Boundary //
    //////////////
    MeshPlyPropertySpec("boundary", "h_negative_B", {"h"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("boundary", "d_through_B", {"d"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("boundary", "rgba_B", {"red", "green", "blue", "alpha"},
                        SampleType::COLOR, false),
    //////////////
    // HalfEdge //
    //////////////
    MeshPlyPropertySpec("half_edge", "v_origin_H", {"v"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "e_undirected_H", {"e"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "f_left_H", {"f"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "c_below_H", {"c"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "h_next_H", {"n"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "h_twin_H", {"t"}, SampleType::INDEX,
                        false),
    MeshPlyPropertySpec("half_edge", "h_flip_H", {"h_flip"}, SampleType::INDEX,
                        false)};

/**
 * @brief Get std::set<std::shared_ptr<MeshPlyPropertySpec>> of all known PLY
 * property specifications.
 *
 */
std::set<std::shared_ptr<MeshPlyPropertySpec>>
get_all_ply_property_specs(std::string ply_property_convention = "MathUtils");

/**
 * @brief Convert mesh samples to PLY mesh samples.
 *
 * @param mesh_samples
 * @return PlyMeshSamples
 */
PlyMeshSamples mesh_to_ply_samples(const MeshSamples &mesh_samples);
/**
 * @brief Convert PLY mesh samples to mesh samples.
 *
 * @param ply_mesh_samples
 * @return MeshSamples
 */
MeshSamples ply_to_mesh_samples(const PlyMeshSamples &ply_mesh_samples);

/**
 * @brief Save PLY mesh samples to a file.
 *
 * @param mesh_samples
 * @param ply_path
 * @param use_binary
 * @param ply_property_convention
 */
void save_ply_samples(const PlyMeshSamples &mesh_samples,
                      const std::string &ply_path, const bool use_binary = true,
                      const std::string &ply_property_convention = "MathUtils");

/**
 * @brief Load PLY mesh samples from a file.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @param ply_property_convention
 * @return PlyMeshSamples
 */
PlyMeshSamples
load_ply_samples(const std::string &filepath,
                 const bool preload_into_memory = true,
                 const bool verbose = false,
                 const std::string &ply_property_convention = "MathUtils");
/**
 * @brief Save mesh samples to a PLY file.
 *
 * @param mesh_samples
 * @param ply_path
 * @param use_binary
 */
void save_mesh_samples(
    const MeshSamples &mesh_samples, const std::string &ply_path,
    const bool use_binary = true,
    const std::string &ply_property_convention = "MathUtils");

/**
 * @brief Load mesh samples from a PLY file.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return MeshSamples
 */
MeshSamples
load_mesh_samples(const std::string &filepath,
                  const bool preload_into_memory = true,
                  const bool verbose = false,
                  const std::string &ply_property_convention = "MathUtils");

} // namespace io
} // namespace mesh
} // namespace mathutils

namespace mathutils {
namespace mesh {
namespace io {

////////////////////////////////////////////
// misc tinyply helpers ////////////////////
////////////////////////////////////////////
/** @addtogroup utils
 *  @{
 */
/**
 * @brief Read a binary file into a vector of bytes.
 *
 * @param pathToFile
 * @return std::vector<uint8_t>
 */
inline std::vector<uint8_t> read_file_binary(const std::string &pathToFile) {
  std::ifstream file(pathToFile, std::ios::binary);
  std::vector<uint8_t> fileBufferBytes;

  if (file.is_open()) {
    file.seekg(0, std::ios::end);
    size_t sizeBytes = file.tellg();
    file.seekg(0, std::ios::beg);
    fileBufferBytes.resize(sizeBytes);
    if (file.read((char *)fileBufferBytes.data(), sizeBytes))
      return fileBufferBytes;
  } else
    throw std::runtime_error("could not open binary ifstream to path " +
                             pathToFile);
  return fileBufferBytes;
}

/**
 * @brief A streambuf that reads from a memory buffer.
 *
 */
struct memory_buffer : public std::streambuf {
  char *p_start{nullptr};
  char *p_end{nullptr};
  size_t size;

  memory_buffer(char const *first_elem, size_t size)
      : p_start(const_cast<char *>(first_elem)), p_end(p_start + size),
        size(size) {
    setg(p_start, p_start, p_end);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (dir == std::ios_base::cur)
      gbump(static_cast<int>(off));
    else
      setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
    return gptr() - p_start;
  }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(pos, std::ios_base::beg, which);
  }
};

/**
 * @brief A stream that reads from a memory buffer.
 *
 */
struct memory_stream : virtual memory_buffer, public std::istream {
  memory_stream(char const *first_elem, size_t size)
      : memory_buffer(first_elem, size),
        std::istream(static_cast<std::streambuf *>(this)) {}
};

/**
 * @brief A timer.
 */
class manual_timer {
  std::chrono::high_resolution_clock::time_point t0;
  double timestamp{0.0};

public:
  void start() { t0 = std::chrono::high_resolution_clock::now(); }
  void stop() {
    timestamp = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0)
                    .count() *
                1000.0;
  }
  const double &get() { return timestamp; }
};
/** @}*/ // end of group utils

////////////////////////////////////////////
// half-edge mesh funs /////////////////////
////////////////////////////////////////////

// /** @addtogroup MeshIO
//  *  @{
//  */
/**
 * @brief loads ply file of vertex-face samples.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return mathutils::VertexFaceTuple
 */
std::pair<Samples3d, Samples3i>
load_vf_samples_from_ply(const std::string &filepath,
                         const bool preload_into_memory = true,
                         const bool verbose = false);

/**
 * @brief writes vertex-face samples to a ply file.
 *
 * @param xyz_coord_V
 * @param V_cycle_F
 * @param ply_path
 * @param use_binary
 */
void write_vf_samples_to_ply(Samples3d &xyz_coord_V, Samples3i &V_cycle_F,
                             const std::string &ply_path,
                             const bool use_binary = true);

/**
 * @brief loads ply file of half-edge samples.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return std::map<std::string, Samplesi>
 */
std::map<std::string, Samplesi>
load_he_samples_from_ply(const std::string &filepath,
                         const bool preload_into_memory = true,
                         const bool verbose = false);

/**
 * @brief writes half-edge samples to a ply file.
 *
 * @param xyz_coord_V
 * ...
 * @param ply_path
 * @param use_binary
 */
void write_he_samples_to_ply(
    const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
    const Samplesi &v_origin_H, const Samplesi &h_next_H,
    const Samplesi &h_twin_H, const Samplesi &f_left_H,
    const Samplesi &h_right_F, const Samplesi &h_negative_B,
    const std::string &ply_path, const bool use_binary = true);

/**
 * @brief loads ply file into map of mesh samples.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return std::map<std::string, std::variant<...>>;
 */
MeshSamples32 load_mesh_samples_from_ply(const std::string &filepath,
                                         const bool preload_into_memory = true,
                                         const bool verbose = false);

/**
 * @brief writes mesh samples to a ply file.
 *
 * @param mesh_samples
 * @param ply_path
 * @param use_binary
 */
void write_mesh_samples_to_ply(const MeshSamples32 &mesh_samples,
                               const std::string &ply_path,
                               const bool use_binary = true);

/** @}*/ // end of group MeshIO

} // namespace io
} // namespace mesh
} // namespace mathutils