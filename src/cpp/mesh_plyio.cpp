/**
 * @file mesh_plyio.cpp
 */

#ifndef TINYPLY_IMPLEMENTATION
#define TINYPLY_IMPLEMENTATION
#endif

#include "mathutils/mesh/mesh_plyio.hpp"
#include "mathutils/io/tinyply.h" // tinyply::PlyFile, tinyply::PlyData
#include <Eigen/Core>             // Eigen::MatrixXd, Eigen::VectorXd
#include <algorithm>              // For std::min and std::max
#include <chrono> // std::chrono::high_resolution_clock and std::chrono::duration
#include <iostream>      // std::cout
#include <set>           // std::set
#include <tuple>         // std::tuple
#include <unordered_set> // std::unordered_set
#include <vector>        // std::vector

namespace mathutils {
namespace mesh_io {

using Samplesi = Eigen::VectorXi;
using Samples2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;
using Samples3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;
using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

//
//

std::pair<Samples3d, Samples3i>
load_vf_samples_from_ply(const std::string &filepath,
                         const bool preload_into_memory, const bool verbose) {
  std::streambuf *oldCoutStreamBuf = nullptr;
  std::ofstream nullStream;

  if (!verbose) {
    // Save the old buffer
    oldCoutStreamBuf = std::cout.rdbuf();

    // Redirect std::cout to /dev/null
    nullStream.open("/dev/null");
    std::cout.rdbuf(nullStream.rdbuf());
  }
  Samples3d xyz_coord_V;
  Samples3i V_cycle_F;
  std::cout << "..............................................................."
               ".........\n";
  std::cout << "Now Reading: " << filepath << std::endl;

  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;
  try {
    // For most files < 1gb, pre-loading the entire file upfront and wrapping it
    // into a stream is a net win for parsing speed, about 40% faster.
    if (preload_into_memory) {
      byte_buffer = read_file_binary(filepath);
      file_stream.reset(
          new memory_stream((char *)byte_buffer.data(), byte_buffer.size()));
    } else {
      file_stream.reset(new std::ifstream(filepath, std::ios::binary));
    }

    if (!file_stream || file_stream->fail())
      throw std::runtime_error("file_stream failed to open " + filepath);

    file_stream->seekg(0, std::ios::end);
    const float size_mb = file_stream->tellg() * float(1e-6);
    file_stream->seekg(0, std::ios::beg);

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    std::cout << "\t[ply_header] Type: "
              << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto &c : file.get_comments())
      std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto &c : file.get_info())
      std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : file.get_elements()) {
      std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
                << std::endl;
      for (const auto &p : e.properties) {
        std::cout << "\t[ply_header] \tproperty: " << p.name
                  << " (type=" << tinyply::PropertyTable[p.propertyType].str
                  << ")";
        if (p.isList)
          std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                    << ")";
        std::cout << std::endl;
      }
    }

    // Because most people have their own mesh types, tinyply treats parsed data
    // as structured/typed byte buffers.
    std::shared_ptr<tinyply::PlyData> vertices, normals, faces;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data.
    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      vertices =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // try {
    //   normals =
    //       file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    // } catch (const std::exception &e) {
    //   std::cerr << "tinyply exception: " << e.what() << std::endl;
    // }

    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    manual_timer read_timer;

    read_timer.start();
    file.read(*file_stream);
    read_timer.stop();

    const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
    std::cout << "\tparsing " << size_mb << "mb in " << parsing_time
              << " seconds [" << (size_mb / parsing_time) << " MBps]"
              << std::endl;

    if (vertices)
      std::cout << "\tRead " << vertices->count << " total vertices "
                << std::endl;
    // if (normals)
    //   std::cout << "\tRead " << normals->count << " total vertex normals "
    //             << std::endl;
    if (faces)
      std::cout << "\tRead " << faces->count << " total faces (triangles) "
                << std::endl;

    // // convert to positions to Samples3d
    // const size_t numVerticesBytes = vertices->buffer.size_bytes();
    // xyz_coord_V.resize(vertices->count, 3);
    // std::memcpy(xyz_coord_V.data(), vertices->buffer.get(),
    // numVerticesBytes);
    // // convert faces to Samples3i
    // const size_t numFacesBytes = faces->buffer.size_bytes();
    // V_cycle_F.resize(faces->count, 3);
    // std::memcpy(V_cycle_F.data(), faces->buffer.get(), numFacesBytes);

    // Convert to positions to Samples3d
    const size_t numVertices = vertices->count;
    xyz_coord_V.resize(numVertices, 3);
    const double *vertexBuffer =
        reinterpret_cast<const double *>(vertices->buffer.get());
    for (size_t i = 0; i < numVertices; ++i) {
      xyz_coord_V(i, 0) = vertexBuffer[3 * i + 0]; // x
      xyz_coord_V(i, 1) = vertexBuffer[3 * i + 1]; // y
      xyz_coord_V(i, 2) = vertexBuffer[3 * i + 2]; // z
    }

    // Convert faces to Samples3i
    const size_t numFaces = faces->count;
    V_cycle_F.resize(numFaces, 3);
    const int *faceBuffer = reinterpret_cast<const int *>(faces->buffer.get());
    for (size_t i = 0; i < numFaces; ++i) {
      V_cycle_F(i, 0) = faceBuffer[3 * i + 0]; // vertex index 1
      V_cycle_F(i, 1) = faceBuffer[3 * i + 1]; // vertex index 2
      V_cycle_F(i, 2) = faceBuffer[3 * i + 2]; // vertex index 3
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return std::make_pair(xyz_coord_V, V_cycle_F);
}

void write_vf_samples_to_ply(Samples3d &xyz_coord_V, Samples3i &V_cycle_F,
                             const std::string &ply_path,
                             const bool use_binary) {

  // std::string ply_path = output_directory + "/" + filename;

  std::filebuf fb;
  fb.open(ply_path,
          use_binary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + ply_path);

  tinyply::PlyFile mesh_file;

  // mesh_file.add_properties_to_element(
  //     "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64, xyz_coord_V.rows(),
  //     reinterpret_cast<uint8_t *>(const_cast<double *>(xyz_coord_V.data())),
  //     tinyply::Type::INVALID, 0);

  // mesh_file.add_properties_to_element(
  //     "face", {"vertex_indices"}, tinyply::Type::INT32, V_cycle_F.rows(),
  //     reinterpret_cast<uint8_t *>(const_cast<int *>(V_cycle_F.data())),
  //     tinyply::Type::UINT8, V_cycle_F.cols());
  // Convert to row-major storage
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      xyz_coord_V_row_major = xyz_coord_V;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      V_cycle_F_row_major = V_cycle_F;

  mesh_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
      xyz_coord_V_row_major.rows(),
      reinterpret_cast<uint8_t *>(xyz_coord_V_row_major.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32,
      V_cycle_F_row_major.rows(),
      reinterpret_cast<uint8_t *>(V_cycle_F_row_major.data()),
      tinyply::Type::UINT8, V_cycle_F_row_major.cols());

  mesh_file.get_comments().push_back("MathUtils vf_ply");

  // Write an ply file
  mesh_file.write(outstream, use_binary);
}

} // namespace mesh_io
} // namespace mathutils