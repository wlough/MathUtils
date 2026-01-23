/**
 * @file mesh_plyio.cpp
 */

#include "mathutils/mesh/mesh_plyio.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include <algorithm>  // For std::min and std::max
#include <chrono> // std::chrono::high_resolution_clock and std::chrono::duration
#include <cstdint>       // std::uint32_t
#include <iostream>      // std::cout
#include <set>           // std::set
#include <tuple>         // std::tuple
#include <typeinfo>      // for debugging typeid
#include <unordered_set> // std::unordered_set
#include <vector>        // std::vector

// #ifndef TINYPLY_IMPLEMENTATION
// #define TINYPLY_IMPLEMENTATION
// #endif
#include "mathutils/io/tinyply.h" // tinyply::PlyFile, tinyply::PlyData

namespace mathutils {
namespace mesh {
namespace io {

// const size_t count = xyz_coord_V_data->count;
// Samples3d xyz_coord_V(count, 3);
// const double *data_ptr = reinterpret_cast<const double
// *>(xyz_coord_V_data->buffer.get()); for (size_t i = 0; i < count; ++i) {
//   for (size_t j = 0; j < 3; ++j) {
//     xyz_coord_V(i, j) = data_ptr[i * 3 + j];
//   }
// }
// mesh_samples["xyz_coord_V"] = xyz_coord_V;

// const size_t count = V_cycle_F_data->count;
// Samples3i V_cycle_F(count, 3);
// const int *data_ptr = reinterpret_cast<const int
// *>(V_cycle_F_data->buffer.get()); for (size_t i = 0; i < count; ++i) {
//   for (size_t j = 0; j < 3; ++j) {
//     V_cycle_F(i, j) = data_ptr[i * 3 + j];
//   }
// }
// mesh_samples["V_cycle_F"] = V_cycle_F;

// Example One: converting to your own application types
// {
//   const size_t numVerticesBytes = vertices->buffer.size_bytes();
//   std::vector<float3> verts(vertices->count);
//   std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
// }

// // Example Two: converting to your own application type
// {
//   std::vector<float3> verts_floats;
//   std::vector<double3> verts_doubles;
//   if (vertices->t == tinyply::Type::FLOAT32) { /* as floats ... */
//   }
//   if (vertices->t == tinyply::Type::FLOAT64) { /* as doubles ... */
//   }
// }

void MeshPlyPropertySpec::add_property_to_mesh_file(
    const PlyMeshSamples &mesh_samples, tinyply::PlyFile &mesh_file) const {
  auto it = mesh_samples.find(samples_key);
  if (it == mesh_samples.end()) {
    std::cerr << "Warning: could not find data for key " << samples_key
              << std::endl;
    return;
  }
  const auto &samples_variant = it->second;

  // try {
  //   const tinyply::Type tinyply_type = PlyTypeFromSampleType.at(sample_type);
  // } catch (const std::out_of_range &e) {
  //   throw std::runtime_error(
  //       "Unsupported PlyTypeFromSampleType.at(sample_type) for key " +
  //       samples_key);
  // }
  const tinyply::Type tinyply_type = PlyTypeFromSampleType.at(sample_type);

  std::visit(
      [&](auto &&samples) {
        using SamplesT = std::decay_t<decltype(samples)>;

        SampleType actual{};
        if constexpr (std::is_same_v<SamplesT, SamplesIndex>)
          actual = SampleType::INDEX;
        else if constexpr (std::is_same_v<SamplesT, SamplesField>)
          actual = SampleType::FIELD;
        else if constexpr (std::is_same_v<SamplesT, SamplesRGBA>)
          actual = SampleType::COLOR;
        else
          throw std::runtime_error("Unsupported sample type for key " +
                                   samples_key);

        if (actual != sample_type) {
          throw std::runtime_error("Sample type mismatch for key " +
                                   samples_key);
        }
        const std::uint32_t Nrows = static_cast<std::uint32_t>(samples.rows());
        auto *data_ptr = reinterpret_cast<std::uint8_t *>(
            const_cast<std::remove_const_t<
                std::remove_pointer_t<decltype(samples.data())>> *>(
                samples.data()));

        if (!is_list) {
          mesh_file.add_properties_to_element(element_key, property_keys,
                                              tinyply_type, Nrows, data_ptr,
                                              tinyply::Type::INVALID, 0);
          return;
        }
        mesh_file.add_properties_to_element(element_key, property_keys,
                                            tinyply_type, Nrows, data_ptr,
                                            tinyply::Type::UINT8, list_count);
      },
      samples_variant);
}

void MeshPlyPropertySpec::add_property_to_mesh_samples(
    PlyMeshSamples &mesh_samples, tinyply::PlyFile &mesh_file) const {
  std::shared_ptr<tinyply::PlyData> samples_ptr =
      mesh_file.request_properties_from_element(element_key, property_keys,
                                                is_list ? list_count : 0);
  const size_t rows = samples_ptr->count;
  std::size_t cols = is_list ? list_count : property_keys.size();
  const size_t numSamplesBytes = samples_ptr->buffer.size_bytes();

  PlySamplesVariant samples_variant;
  if (sample_type == SampleType::INDEX) {
    mesh_samples[samples_key] = Matrix<PlyIndex>(rows, cols);
  } else if (sample_type == SampleType::FIELD) {
    mesh_samples[samples_key] = Matrix<PlyReal>(rows, cols);
  } else if (sample_type == SampleType::COLOR) {
    mesh_samples[samples_key] = Matrix<PlyColor>(rows, cols);
  } else {
    throw std::runtime_error("Unsupported sample type for key " + samples_key);
  }

  std::visit(
      [&](auto &&samples) {
        std::memcpy(samples.data(), samples_ptr->buffer.get(), numSamplesBytes);
      },
      mesh_samples[samples_key]);
}

std::shared_ptr<tinyply::PlyData>
MeshPlyPropertySpec::request_property_from_mesh_file(
    tinyply::PlyFile &mesh_file) const {
  return mesh_file.request_properties_from_element(element_key, property_keys,
                                                   is_list ? list_count : 0);
}

PlyMeshSamples mesh_to_ply_samples(const MeshSamples &mesh_samples) {

  PlyMeshSamples ply_samples;

  for (const auto &[key, value] : mesh_samples) {
    std::visit(
        [&](const auto &s) {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, SamplesIndex>) {
            ply_samples.insert_or_assign(key, s.template to_dtype<PlyIndex>());
          } else if constexpr (std::is_same_v<T, SamplesField>) {
            ply_samples.insert_or_assign(key, s.template to_dtype<PlyReal>());
          } else if constexpr (std::is_same_v<T, SamplesRGBA>) {
            ply_samples.insert_or_assign(key, s.template to_dtype<PlyColor>());
          } else {
            throw std::runtime_error(
                "to_ply_mesh_samples: unsupported sample variant type " +
                std::string(typeid(T).name()) + " for key " + key);
          }
        },
        value);
  }
  return ply_samples;
}

MeshSamples ply_to_mesh_samples(const PlyMeshSamples &ply_mesh_samples) {

  MeshSamples mesh_samples;

  for (const auto &[key, value] : ply_mesh_samples) {
    std::visit(
        [&](const auto &s) {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, PlySamplesIndex>) {
            mesh_samples.insert_or_assign(key, s.template to_dtype<Index>());
          } else if constexpr (std::is_same_v<T, PlySamplesField>) {
            mesh_samples.insert_or_assign(key, s.template to_dtype<Real>());
          } else if constexpr (std::is_same_v<T, PlySamplesRGBA>) {
            mesh_samples.insert_or_assign(key, s.template to_dtype<Color>());
          } else {
            throw std::runtime_error(
                "from_ply_mesh_samples: unsupported sample variant type " +
                std::string(typeid(T).name()) + " for key " + key);
          }
        },
        value);
  }
  return mesh_samples;
}

void save_ply_samples(const PlyMeshSamples &mesh_samples,
                      const std::string &ply_path, const bool use_binary) {
  std::filebuf fb;
  fb.open(ply_path,
          use_binary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + ply_path);
  tinyply::PlyFile mesh_file;
  for (const auto &property_spec : PlyPropertySpecs) {
    try {
      property_spec.add_property_to_mesh_file(mesh_samples, mesh_file);
    } catch (const std::exception &e) {
      std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
  }
  mesh_file.get_comments().push_back("MathUtils ply");
  mesh_file.write(outstream, use_binary);
}

PlyMeshSamples load_ply_samples(const std::string &filepath,
                                const bool preload_into_memory,
                                const bool verbose) {

  PlyMeshSamples mesh_samples;

  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;
  try {
    if (preload_into_memory) {
      byte_buffer = read_file_binary(filepath);
      file_stream.reset(
          new memory_stream((char *)byte_buffer.data(), byte_buffer.size()));
    } else {
      file_stream.reset(new std::ifstream(filepath, std::ios::binary));
    }
    if (!file_stream || file_stream->fail()) {
      throw std::runtime_error("file_stream failed to open " + filepath);
    }

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    if (verbose) {
      std::cout << "\t[ply_header] Type: "
                << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
      for (const auto &c : file.get_comments()) {
        std::cout << "\t[ply_header] Comment: " << c << std::endl;
      }
      for (const auto &c : file.get_info()) {
        std::cout << "\t[ply_header] Info: " << c << std::endl;
      }
    }

    std::set<std::string> element_names;
    for (const auto &e : file.get_elements()) {
      std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
                << std::endl;
      element_names.insert(e.name);
    }

    std::map<std::string, std::shared_ptr<tinyply::PlyData>> requested_data;
    for (const auto &property_spec : PlyPropertySpecs) {
      if (element_names.find(property_spec.element_key) ==
          element_names.end()) {
        if (verbose) {
          std::cout << "\t[ply_load] skipping property key "
                    << property_spec.samples_key << " because element "
                    << property_spec.element_key << " not found in file."
                    << std::endl;
        }
        continue;
      }
      try {
        // property_spec.add_property_to_mesh_samples(mesh_samples, file);
        requested_data[property_spec.samples_key] =
            property_spec.request_property_from_mesh_file(file);
      } catch (const std::exception &e) {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
      }
    }

    file.read(*file_stream);

    for (const auto &[key, samples_ptr] : requested_data) {

      // try {
      MeshPlyPropertySpec property_spec = PlyPropertyTable.at(key);
      // } catch (const std::out_of_range &e) {
      //   throw std::runtime_error(
      //       "Unsupported PlyPropertyTable.at(key) for key " + key);
      // }

      const size_t rows = samples_ptr->count;
      std::size_t cols = property_spec.is_list
                             ? property_spec.list_count
                             : property_spec.property_keys.size();
      const size_t numSamplesBytes = samples_ptr->buffer.size_bytes();

      PlySamplesVariant samples_variant;
      if (property_spec.sample_type == SampleType::INDEX) {
        mesh_samples[property_spec.samples_key] = Matrix<PlyIndex>(rows, cols);
      } else if (property_spec.sample_type == SampleType::FIELD) {
        mesh_samples[property_spec.samples_key] = Matrix<PlyReal>(rows, cols);
      } else if (property_spec.sample_type == SampleType::COLOR) {
        mesh_samples[property_spec.samples_key] = Matrix<PlyColor>(rows, cols);
      } else {
        throw std::runtime_error("Unsupported sample type for key " +
                                 property_spec.samples_key);
      }

      std::visit(
          [&](auto &&samples) {
            std::memcpy(samples.data(), samples_ptr->buffer.get(),
                        numSamplesBytes);
          },
          mesh_samples[property_spec.samples_key]);
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }
  return mesh_samples;
}

void save_mesh_samples(const MeshSamples &mesh_samples,
                       const std::string &ply_path, const bool use_binary) {
  PlyMeshSamples ply_samples = mesh_to_ply_samples(mesh_samples);
  save_ply_samples(ply_samples, ply_path, use_binary);
}

MeshSamples load_mesh_samples(const std::string &filepath,
                              const bool preload_into_memory,
                              const bool verbose) {
  PlyMeshSamples ply_samples =
      load_ply_samples(filepath, preload_into_memory, verbose);

  return ply_to_mesh_samples(ply_samples);
}
} // namespace io
} // namespace mesh
} // namespace mathutils

namespace mathutils {
namespace mesh {
namespace io {

// template class for ply element names and keys
template <typename InputDataType, typename StorageDataType,
          tinyply::Type TinyplyDataType, int dim, bool is_list>
struct PlyPropertySamplesTemplate {

  static_assert(!is_list || (dim >= 1 && dim <= 255),
                "list_count must fit in uint8_t and be >= 1.");
  static_assert(dim != Eigen::Dynamic, "This template requires fixed dim.");

  using InputSamples =
      Eigen::Matrix<InputDataType, Eigen::Dynamic, dim,
                    (dim == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using StorageSamples =
      Eigen::Matrix<StorageDataType, Eigen::Dynamic, dim,
                    (dim == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;

  std::string mesh_samples_key = "";
  // std::string property_name = "";
  std::string element_name = "";
  std::vector<std::string> property_names;
  bool skip = false;
  static constexpr bool need_to_convert_dtype =
      !std::is_same_v<InputDataType, StorageDataType>;

  // instances of PlyPropertySamplesTemplate
  // will live in a vector inside the write_mesh_samples_to_ply
  // function scope pointers to samples_storage will remain valid
  // until mesh_file.write(...) is called inside that function
  StorageSamples samples_storage;

  // Constructor for single-name property (scalar or list)
  PlyPropertySamplesTemplate(std::string mesh_samples_key_,
                             std::string property_name_,
                             std::string element_name_)
      : mesh_samples_key(std::move(mesh_samples_key_)),
        element_name(std::move(element_name_)),
        property_names{std::move(property_name_)} {}

  // Constructor for multiple scalar properties (e.g., "x", "y", "z" stored
  // as
  // columns of a single SamplesType)
  PlyPropertySamplesTemplate(std::string mesh_samples_key_,
                             std::vector<std::string> property_names_,
                             std::string element_name_)
      : mesh_samples_key(std::move(mesh_samples_key_)),
        element_name(std::move(element_name_)),
        property_names(std::move(property_names_)) {}

  void add_property_to_mesh_file(const MeshSamples &mesh_samples,
                                 tinyply::PlyFile &mesh_file) {
    skip = false;

    // Validate naming scheme up-front (compile-time branches)
    if constexpr (is_list) {
      if (property_names.size() != 1) {
        skip = true;
        std::cerr << "Warning: only one property_name is supported for "
                     "lists but key "
                  << mesh_samples_key << " has " << property_names.size()
                  << "\n";
        return;
      }
    } else {
      if (property_names.size() != static_cast<size_t>(dim)) {
        skip = true;
        std::cerr << "Warning: property_names size mismatch for key "
                  << mesh_samples_key << ": expected " << dim
                  << " names but got " << property_names.size() << "\n";
        return;
      }
    }

    auto it = mesh_samples.find(mesh_samples_key);
    if (it == mesh_samples.end()) {
      skip = true;
      std::cerr << "Warning: could not find data for key " << mesh_samples_key
                << std::endl;
      return;
    }
    const InputSamples *samples_ptr = std::get_if<InputSamples>(&it->second);
    if (!samples_ptr) {
      skip = true;
      std::cerr << "Warning: could not find data for key " << mesh_samples_key
                << " with expected type " << typeid(InputSamples).name()
                << std::endl;
      return;
    }
    if (samples_ptr->cols() != dim) {
      skip = true;
      std::cerr << "Warning: dimension mismatch for key " << mesh_samples_key
                << ": expected " << dim << " but got " << samples_ptr->cols()
                << std::endl;
      return;
    }
    if (samples_ptr->size() == 0) {
      skip = true;
      std::cerr << "Warning: zero size for key " << mesh_samples_key
                << std::endl;
      return;
    }

    const auto rows = static_cast<uint32_t>(samples_ptr->rows());

    auto add_from_ptr = [&](void *data_ptr) {
      if constexpr (dim == 1 && !is_list) {
        mesh_file.add_properties_to_element(
            element_name, property_names, TinyplyDataType, rows,
            reinterpret_cast<uint8_t *>(data_ptr), tinyply::Type::INVALID, 0);
      } else if constexpr (is_list) {
        mesh_file.add_properties_to_element(
            element_name, property_names, TinyplyDataType, rows,
            reinterpret_cast<uint8_t *>(data_ptr), tinyply::Type::UINT8,
            static_cast<uint32_t>(dim));
      } else {
        mesh_file.add_properties_to_element(
            element_name, property_names, TinyplyDataType, rows,
            reinterpret_cast<uint8_t *>(data_ptr), tinyply::Type::INVALID, 0);
      }
    };

    if constexpr (need_to_convert_dtype) {
      samples_storage = samples_ptr->template cast<StorageDataType>();
      add_from_ptr(static_cast<void *>(samples_storage.data()));
    } else {
      add_from_ptr(static_cast<void *>(
          const_cast<InputDataType *>(samples_ptr->data())));
    }
  }
};

} // namespace io
} // namespace mesh
} // namespace mathutils

namespace mathutils {
namespace mesh {
namespace io {

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

MeshSamples32 load_mesh_samples_from_ply(const std::string &filepath,
                                         const bool preload_into_memory,
                                         const bool verbose) {
  using Samplesi = SamplesTypeDimTemplate<std::int32_t, 1>;
  using Samples2i = SamplesTypeDimTemplate<std::int32_t, 2>;
  using Samples3i = SamplesTypeDimTemplate<std::int32_t, 3>;
  using Samples4i = SamplesTypeDimTemplate<std::int32_t, 4>;

  std::streambuf *oldCoutStreamBuf = nullptr;
  std::ofstream nullStream;

  if (!verbose) {
    // Save the old buffer
    oldCoutStreamBuf = std::cout.rdbuf();

    // Redirect std::cout to /dev/null
    nullStream.open("/dev/null");
    std::cout.rdbuf(nullStream.rdbuf());
  }

  MeshSamples32 mesh_samples;

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

    if (!file_stream || file_stream->fail()) {
      throw std::runtime_error("file_stream failed to open " + filepath);
    }

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    std::cout << "\t[ply_header] Type: "
              << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto &c : file.get_comments()) {
      std::cout << "\t[ply_header] Comment: " << c << std::endl;
    }
    for (const auto &c : file.get_info()) {
      std::cout << "\t[ply_header] Info: " << c << std::endl;
    }
    std::vector<std::string> element_names;
    std::vector<std::vector<std::string>> property_names_per_element;

    std::vector<std::string> vertex_property_names;
    bool has_vertex_element = false;
    bool load_xyz_coord_V = false;
    std::shared_ptr<tinyply::PlyData> xyz_coord_V_data;
    bool load_quat_frame_V = false;
    std::shared_ptr<tinyply::PlyData> quat_frame_V_data;
    bool load_h_out_V = false;
    std::shared_ptr<tinyply::PlyData> h_out_V_data;
    bool load_d_through_V = false;
    std::shared_ptr<tinyply::PlyData> d_through_V_data;
    bool load_rgba_V = false; // all 4 color channels must exist
    std::shared_ptr<tinyply::PlyData> rgba_V_data;

    std::vector<std::string> edge_property_names;
    bool has_edge_element = false;
    bool load_V_cycle_E = false;
    std::shared_ptr<tinyply::PlyData> V_cycle_E_data;
    bool load_h_directed_E = false;
    std::shared_ptr<tinyply::PlyData> h_directed_E_data;
    bool load_d_through_E = false;
    std::shared_ptr<tinyply::PlyData> d_through_E_data;

    std::vector<std::string> face_property_names;
    bool has_face_element = false;
    bool load_V_cycle_F = false;
    std::shared_ptr<tinyply::PlyData> V_cycle_F_data;
    bool load_h_right_F = false;
    std::shared_ptr<tinyply::PlyData> h_right_F_data;
    bool load_d_through_F = false;
    std::shared_ptr<tinyply::PlyData> d_through_F_data;

    std::vector<std::string> cell_property_names;
    bool has_cell_element = false;
    bool load_V_cycle_C = false;
    std::shared_ptr<tinyply::PlyData> V_cycle_C_data;
    bool load_h_above_C = false;
    std::shared_ptr<tinyply::PlyData> h_above_C_data;
    bool load_d_through_C = false;
    std::shared_ptr<tinyply::PlyData> d_through_C_data;

    std::vector<std::string> boundary_property_names;
    bool has_boundary_element = false;
    bool load_h_negative_B = false;
    std::shared_ptr<tinyply::PlyData> h_negative_B_data;
    bool load_d_through_B = false;
    std::shared_ptr<tinyply::PlyData> d_through_B_data;

    std::vector<std::string> halfedge_property_names;
    bool has_halfedge_element = false;
    bool load_v_origin_H = false;
    std::shared_ptr<tinyply::PlyData> v_origin_H_data;
    bool load_e_undirected_H = false;
    std::shared_ptr<tinyply::PlyData> e_undirected_H_data;
    bool load_f_left_H = false;
    std::shared_ptr<tinyply::PlyData> f_left_H_data;
    bool load_c_below_H = false;
    std::shared_ptr<tinyply::PlyData> c_below_H_data;
    bool load_h_next_H = false;
    std::shared_ptr<tinyply::PlyData> h_next_H_data;
    bool load_h_twin_H = false;
    std::shared_ptr<tinyply::PlyData> h_twin_H_data;
    bool load_h_flip_H = false;
    std::shared_ptr<tinyply::PlyData> h_flip_H_data;

    std::vector<std::string> dart_property_names;
    bool has_dart_element = false;
    bool load_v_in_D = false;
    std::shared_ptr<tinyply::PlyData> v_in_D_data;
    bool load_e_in_D = false;
    std::shared_ptr<tinyply::PlyData> e_in_D_data;
    bool load_f_in_D = false;
    std::shared_ptr<tinyply::PlyData> f_in_D_data;
    bool load_c_in_D = false;
    std::shared_ptr<tinyply::PlyData> c_in_D_data;
    bool load_d_diff0_D = false;
    std::shared_ptr<tinyply::PlyData> d_diff0_D_data;
    bool load_d_diff1_D = false;
    std::shared_ptr<tinyply::PlyData> d_diff1_D_data;
    bool load_d_diff2_D = false;
    std::shared_ptr<tinyply::PlyData> d_diff2_D_data;
    bool load_d_diff3_D = false;
    std::shared_ptr<tinyply::PlyData> d_diff3_D_data;

    for (const auto &e : file.get_elements()) {
      std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
                << std::endl;
      element_names.push_back(e.name);
      std::vector<std::string> property_names;
      for (const auto &p : e.properties) {
        std::cout << "\t[ply_header] \tproperty: " << p.name
                  << " (type=" << tinyply::PropertyTable[p.propertyType].str
                  << ")";
        property_names.push_back(p.name);
        if (p.isList) {
          std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                    << ")";
          std::cout << std::endl;
        }
      }
      property_names_per_element.push_back(property_names);
      if (e.name == "vertex") {
        has_vertex_element = true;
        vertex_property_names = property_names;
      } else if (e.name == "edge") {
        has_edge_element = true;
        edge_property_names = property_names;
      } else if (e.name == "face") {
        has_face_element = true;
        face_property_names = property_names;
      } else if (e.name == "cell") {
        has_cell_element = true;
        cell_property_names = property_names;
      } else if (e.name == "boundary") {
        has_boundary_element = true;
        boundary_property_names = property_names;
      } else if (e.name == "half_edge") {
        has_halfedge_element = true;
        halfedge_property_names = property_names;
      } else if (e.name == "dart") {
        has_dart_element = true;
        dart_property_names = property_names;
      }
    }

    ///////////////////////////////////////////
    // Vertices
    ///////////////////////////////////////////
    if (has_vertex_element) {
      bool has_x = false;
      bool has_y = false;
      bool has_z = false;
      bool has_qw = false, has_qx = false, has_qy = false, has_qz = false;
      bool has_h_out = false, has_h = false; // h for backward compatibility
      bool has_d_through = false;
      bool has_red = false, has_green = false, has_blue = false,
           has_alpha = false;
      for (const auto &prop_name : vertex_property_names) {
        if (prop_name == "x") {
          has_x = true;
        } else if (prop_name == "y") {
          has_y = true;
        } else if (prop_name == "z") {
          has_z = true;
        } else if (prop_name == "h_out") {
          has_h_out = true;
        } else if (prop_name == "h") {
          has_h = true;
        } else if (prop_name == "d_through") {
          has_d_through = true;
        } else if (prop_name == "red") {
          has_red = true;
        } else if (prop_name == "green") {
          has_green = true;
        } else if (prop_name == "blue") {
          has_blue = true;
        } else if (prop_name == "alpha") {
          has_alpha = true;
        } else if (prop_name == "qw") {
          has_qw = true;
        } else if (prop_name == "qx") {
          has_qx = true;
        } else if (prop_name == "qy") {
          has_qy = true;
        } else if (prop_name == "qz") {
          has_qz = true;
        }
      }

      load_xyz_coord_V = has_x && has_y && has_z;
      load_quat_frame_V = has_qw && has_qx && has_qy && has_qz;
      load_h_out_V = has_h_out || has_h;
      load_d_through_V = has_d_through;
      load_rgba_V = has_red && has_green && has_blue &&
                    has_alpha; // all 4 color channels must exist

      if (load_xyz_coord_V) {
        xyz_coord_V_data =
            file.request_properties_from_element("vertex", {"x", "y", "z"});
      }
      if (load_quat_frame_V) {
        quat_frame_V_data = file.request_properties_from_element(
            "vertex", {"qw", "qx", "qy", "qz"});
      }
      if (load_h_out_V) {
        try {
          h_out_V_data =
              file.request_properties_from_element("vertex", {"h_out"});
        } catch (const std::exception &e) {
          // backward compatibility: try to load "h" property
          h_out_V_data = file.request_properties_from_element("vertex", {"h"});
        }
      }
      if (load_d_through_V) {
        d_through_V_data =
            file.request_properties_from_element("vertex", {"d_through"});
      }
    }

    ///////////////////////////////////////////
    // Edges
    ///////////////////////////////////////////
    if (has_edge_element) {
      bool has_vertex_indices = false;
      bool has_h_directed = false;
      bool has_d_through = false;

      for (const auto &prop_name : edge_property_names) {
        if (prop_name == "vertex_indices") {
          has_vertex_indices = true;
        } else if (prop_name == "h_directed") {
          has_h_directed = true;
        } else if (prop_name == "d_through") {
          has_d_through = true;
        }
      }

      load_V_cycle_E = has_vertex_indices;
      load_h_directed_E = has_h_directed;
      load_d_through_E = has_d_through;

      if (load_V_cycle_E) {
        V_cycle_E_data =
            file.request_properties_from_element("edge", {"vertex_indices"});
      }
      if (load_h_directed_E) {
        h_directed_E_data =
            file.request_properties_from_element("edge", {"h_directed"});
      }
      if (load_d_through_E) {
        d_through_E_data =
            file.request_properties_from_element("edge", {"d_through"});
      }
    }

    ///////////////////////////////////////////
    // Faces
    ///////////////////////////////////////////
    if (has_face_element) {
      bool has_vertex_indices = false;
      bool has_h_right = false, has_h = false; // h for backward compatibility
      bool has_d_through = false;

      for (const auto &prop_name : face_property_names) {
        if (prop_name == "vertex_indices") {
          has_vertex_indices = true;
        } else if (prop_name == "h_right") {
          has_h_right = true;
        } else if (prop_name == "h") {
          has_h = true;
        } else if (prop_name == "d_through") {
          has_d_through = true;
        }
      }

      load_V_cycle_F = has_vertex_indices;
      load_h_right_F = has_h_right || has_h;
      load_d_through_F = has_d_through;

      if (load_V_cycle_F) {
        V_cycle_F_data =
            file.request_properties_from_element("face", {"vertex_indices"});
      }
      if (load_h_right_F) {
        try {
          h_right_F_data =
              file.request_properties_from_element("face", {"h_right"});
        } catch (const std::exception &e) {
          // backward compatibility: try to load "h" property
          h_right_F_data = file.request_properties_from_element("face", {"h"});
        }
      }
      if (load_d_through_F) {
        d_through_F_data =
            file.request_properties_from_element("face", {"d_through"});
      }
    }

    ///////////////////////////////////////////
    // Cells
    ///////////////////////////////////////////
    if (has_cell_element) {
      bool has_vertex_indices = false;
      bool has_h_above = false;
      bool has_d_through = false;

      for (const auto &prop_name : cell_property_names) {
        if (prop_name == "vertex_indices") {
          has_vertex_indices = true;
        } else if (prop_name == "h_above") {
          has_h_above = true;
        } else if (prop_name == "d_through") {
          has_d_through = true;
        }
      }

      load_V_cycle_C = has_vertex_indices;
      load_h_above_C = has_h_above;
      load_d_through_C = has_d_through;

      if (load_V_cycle_C) {
        V_cycle_C_data =
            file.request_properties_from_element("cell", {"vertex_indices"});
      }
      if (load_h_above_C) {
        h_above_C_data =
            file.request_properties_from_element("cell", {"h_above"});
      }
      if (load_d_through_C) {
        d_through_C_data =
            file.request_properties_from_element("cell", {"d_through"});
      }
    }

    ///////////////////////////////////////////
    // Boundaries
    ///////////////////////////////////////////
    if (has_boundary_element) {
      bool has_h_negative = false,
           has_h = false; // h for backward compatibility
      bool has_d_through = false;

      for (const auto &prop_name : boundary_property_names) {
        if (prop_name == "h_negative") {
          has_h_negative = true;
        } else if (prop_name == "h") {
          has_h = true;
        } else if (prop_name == "d_through") {
          has_d_through = true;
        }
      }

      load_h_negative_B = has_h_negative || has_h;
      load_d_through_B = has_d_through;

      if (load_h_negative_B) {
        try {
          h_negative_B_data =
              file.request_properties_from_element("boundary", {"h_negative"});
        } catch (const std::exception &e) {
          std::cerr << "tinyply exception: " << e.what()
                    << ". Trying to load 'h' property for boundary instead. "
                    << std::endl;
          try {
            // backward compatibility: try to load "h" property
            std::cout
                << "\t[ply_header] trying to load 'h' property for boundary"
                << std::endl;
            printf(
                "\tf[ply_header] trying to load 'h' property for boundary\n");
            h_negative_B_data =
                file.request_properties_from_element("boundary", {"h"});
          } catch (const std::exception &e) {
            std::cerr << "tinyply exception: " << e.what() << std::endl;
          }
        }
      }
      if (load_d_through_B) {
        d_through_B_data =
            file.request_properties_from_element("boundary", {"d_through"});
      }
    }

    ///////////////////////////////////////////
    // Half-edges
    ///////////////////////////////////////////
    if (has_halfedge_element) {
      bool has_v_origin = false, has_v = false; // v for backward compatibility
      bool has_e_undirected = false;
      bool has_f_left = false, has_f = false; // f for backward compatibility
      bool has_c_below = false;
      bool has_h_next = false, has_n = false; // n for backward compatibility
      bool has_h_twin = false, has_t = false; // t for backward compatibility
      bool has_h_flip = false;

      for (const auto &prop_name : halfedge_property_names) {
        if (prop_name == "v_origin") {
          has_v_origin = true;
        } else if (prop_name == "v") {
          has_v = true;
        } else if (prop_name == "e_undirected") {
          has_e_undirected = true;
        } else if (prop_name == "f_left") {
          has_f_left = true;
        } else if (prop_name == "f") {
          has_f = true;
        } else if (prop_name == "c_below") {
          has_c_below = true;
        } else if (prop_name == "h_next") {
          has_h_next = true;
        } else if (prop_name == "n") {
          has_n = true;
        } else if (prop_name == "h_twin") {
          has_h_twin = true;
        } else if (prop_name == "t") {
          has_t = true;
        } else if (prop_name == "h_flip") {
          has_h_flip = true;
        }
      }

      load_v_origin_H = has_v_origin || has_v;
      load_e_undirected_H = has_e_undirected;
      load_f_left_H = has_f_left || has_f;
      load_c_below_H = has_c_below;
      load_h_next_H = has_h_next || has_n;
      load_h_twin_H = has_h_twin || has_t;
      load_h_flip_H = has_h_flip;

      if (load_v_origin_H) {
        try {
          v_origin_H_data =
              file.request_properties_from_element("half_edge", {"v_origin"});
        } catch (const std::exception &e) {
          // backward compatibility: try to load "v" property
          v_origin_H_data =
              file.request_properties_from_element("half_edge", {"v"});
        }
      }
      if (load_e_undirected_H) {
        e_undirected_H_data =
            file.request_properties_from_element("half_edge", {"e_undirected"});
      }
      if (load_f_left_H) {
        try {
          f_left_H_data =
              file.request_properties_from_element("half_edge", {"f_left"});
        } catch (const std::exception &e) {
          // backward compatibility: try to load "f" property
          f_left_H_data =
              file.request_properties_from_element("half_edge", {"f"});
        }
      }
      if (load_c_below_H) {
        c_below_H_data =
            file.request_properties_from_element("half_edge", {"c_below"});
      }
      if (load_h_next_H) {
        try {
          h_next_H_data =
              file.request_properties_from_element("half_edge", {"h_next"});
        } catch (const std::exception &e) {
          // backward compatibility: try to load "n" property
          h_next_H_data =
              file.request_properties_from_element("half_edge", {"n"});
        }
      }
      if (load_h_twin_H) {
        try {
          h_twin_H_data =
              file.request_properties_from_element("half_edge", {"h_twin"});
        } catch (const std::exception &e) {
          // backward compatibility: try to load "t" property
          h_twin_H_data =
              file.request_properties_from_element("half_edge", {"t"});
        }
      }
      if (load_h_flip_H) {
        h_flip_H_data =
            file.request_properties_from_element("half_edge", {"h_flip"});
      }
    }

    ///////////////////////////////////////////
    // Darts
    ///////////////////////////////////////////
    if (has_dart_element) {
      bool has_v_in = false;
      bool has_e_in = false;
      bool has_f_in = false;
      bool has_c_in = false;
      bool has_d_diff0 = false;
      bool has_d_diff1 = false;
      bool has_d_diff2 = false;
      bool has_d_diff3 = false;

      for (const auto &prop_name : dart_property_names) {
        if (prop_name == "v_in") {
          has_v_in = true;
        } else if (prop_name == "e_in") {
          has_e_in = true;
        } else if (prop_name == "f_in") {
          has_f_in = true;
        } else if (prop_name == "c_in") {
          has_c_in = true;
        } else if (prop_name == "d_diff0") {
          has_d_diff0 = true;
        } else if (prop_name == "d_diff1") {
          has_d_diff1 = true;
        } else if (prop_name == "d_diff2") {
          has_d_diff2 = true;
        } else if (prop_name == "d_diff3") {
          has_d_diff3 = true;
        }
      }

      load_v_in_D = has_v_in;
      load_e_in_D = has_e_in;
      load_f_in_D = has_f_in;
      load_c_in_D = has_c_in;
      load_d_diff0_D = has_d_diff0;
      load_d_diff1_D = has_d_diff1;
      load_d_diff2_D = has_d_diff2;
      load_d_diff3_D = has_d_diff3;

      if (load_v_in_D) {
        v_in_D_data = file.request_properties_from_element("dart", {"v_in"});
      }
      if (load_e_in_D) {
        e_in_D_data = file.request_properties_from_element("dart", {"e_in"});
      }
      if (load_f_in_D) {
        f_in_D_data = file.request_properties_from_element("dart", {"f_in"});
      }
      if (load_c_in_D) {
        c_in_D_data = file.request_properties_from_element("dart", {"c_in"});
      }
      if (load_d_diff0_D) {
        d_diff0_D_data =
            file.request_properties_from_element("dart", {"d_diff0"});
      }
      if (load_d_diff1_D) {
        d_diff1_D_data =
            file.request_properties_from_element("dart", {"d_diff1"});
      }
      if (load_d_diff2_D) {
        d_diff2_D_data =
            file.request_properties_from_element("dart", {"d_diff2"});
      }
      if (load_d_diff3_D) {
        d_diff3_D_data =
            file.request_properties_from_element("dart", {"d_diff3"});
      }
    }

    file.read(*file_stream);

    ///////////////////////////////////////////
    // Vertices
    ///////////////////////////////////////////
    if (load_xyz_coord_V) {
      const size_t count = xyz_coord_V_data->count;
      Samples3d xyz_coord_V(count, 3);
      const double *data_ptr =
          reinterpret_cast<const double *>(xyz_coord_V_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          xyz_coord_V(i, j) = data_ptr[i * 3 + j];
        }
      }
      mesh_samples["xyz_coord_V"] = xyz_coord_V;
    }
    if (load_quat_frame_V) {
      const size_t count = quat_frame_V_data->count;
      Samples4d quat_frame_V(count, 4);
      const double *data_ptr =
          reinterpret_cast<const double *>(quat_frame_V_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          quat_frame_V(i, j) = data_ptr[i * 4 + j];
        }
      }
      mesh_samples["quat_frame_V"] = quat_frame_V;
    }
    if (load_h_out_V) {
      const size_t count = h_out_V_data->count;
      Samplesi h_out_V(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_out_V_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_out_V(i, 0) = data_ptr[i];
      }
      mesh_samples["h_out_V"] = h_out_V;
    }
    if (load_d_through_V) {
      const size_t count = d_through_V_data->count;
      Samplesi d_through_V(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_through_V_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_through_V(i, 0) = data_ptr[i];
      }
      mesh_samples["d_through_V"] = d_through_V;
    }
    if (load_rgba_V) {
      const size_t count = rgba_V_data->count;
      Samples4i rgba_V(count, 4);
      const uint8_t *data_ptr =
          reinterpret_cast<const uint8_t *>(rgba_V_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          rgba_V(i, j) = static_cast<int>(data_ptr[i * 4 + j]);
        }
      }
      mesh_samples["rgba_V"] = rgba_V;
    }

    ///////////////////////////////////////////
    // Edges
    ///////////////////////////////////////////
    if (load_V_cycle_E) {
      const size_t count = V_cycle_E_data->count;
      Samples2i V_cycle_E(count, 2);
      const int *data_ptr =
          reinterpret_cast<const int *>(V_cycle_E_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          V_cycle_E(i, j) = data_ptr[i * 2 + j];
        }
      }
      mesh_samples["V_cycle_E"] = V_cycle_E;
    }
    if (load_h_directed_E) {
      const size_t count = h_directed_E_data->count;
      Samplesi h_directed_E(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_directed_E_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_directed_E(i, 0) = data_ptr[i];
      }
      mesh_samples["h_directed_E"] = h_directed_E;
    }
    if (load_d_through_E) {
      const size_t count = d_through_E_data->count;
      Samplesi d_through_E(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_through_E_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_through_E(i, 0) = data_ptr[i];
      }
      mesh_samples["d_through_E"] = d_through_E;
    }

    ///////////////////////////////////////////
    // Faces
    ///////////////////////////////////////////
    if (load_V_cycle_F) {
      const size_t count = V_cycle_F_data->count;
      Samples3i V_cycle_F(count, 3);
      const int *data_ptr =
          reinterpret_cast<const int *>(V_cycle_F_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          V_cycle_F(i, j) = data_ptr[i * 3 + j];
        }
      }
      mesh_samples["V_cycle_F"] = V_cycle_F;
    }
    if (load_h_right_F) {
      const size_t count = h_right_F_data->count;
      Samplesi h_right_F(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_right_F_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_right_F(i, 0) = data_ptr[i];
      }
      mesh_samples["h_right_F"] = h_right_F;
    }
    if (load_d_through_F) {
      const size_t count = d_through_F_data->count;
      Samplesi d_through_F(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_through_F_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_through_F(i, 0) = data_ptr[i];
      }
      mesh_samples["d_through_F"] = d_through_F;
    }

    ///////////////////////////////////////////
    // Cells
    ///////////////////////////////////////////
    if (load_V_cycle_C) {
      const size_t count = V_cycle_C_data->count;
      Samples4i V_cycle_C(count, 4);
      const int *data_ptr =
          reinterpret_cast<const int *>(V_cycle_C_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          V_cycle_C(i, j) = data_ptr[i * 4 + j];
        }
      }
      mesh_samples["V_cycle_C"] = V_cycle_C;
    }
    if (load_h_above_C) {
      const size_t count = h_above_C_data->count;
      Samplesi h_above_C(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_above_C_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_above_C(i, 0) = data_ptr[i];
      }
      mesh_samples["h_above_C"] = h_above_C;
    }
    if (load_d_through_C) {
      const size_t count = d_through_C_data->count;
      Samplesi d_through_C(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_through_C_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_through_C(i, 0) = data_ptr[i];
      }
      mesh_samples["d_through_C"] = d_through_C;
    }

    ///////////////////////////////////////////
    // Boundaries
    ///////////////////////////////////////////
    if (load_h_negative_B) {
      const size_t count = h_negative_B_data->count;
      Samplesi h_negative_B(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_negative_B_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_negative_B(i, 0) = data_ptr[i];
      }
      mesh_samples["h_negative_B"] = h_negative_B;
    }
    if (load_d_through_B) {
      const size_t count = d_through_B_data->count;
      Samplesi d_through_B(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_through_B_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_through_B(i, 0) = data_ptr[i];
      }
      mesh_samples["d_through_B"] = d_through_B;
    }

    ///////////////////////////////////////////
    // Half-edges
    ///////////////////////////////////////////
    if (load_v_origin_H) {
      const size_t count = v_origin_H_data->count;
      Samplesi v_origin_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(v_origin_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        v_origin_H(i, 0) = data_ptr[i];
      }
      mesh_samples["v_origin_H"] = v_origin_H;
    }
    if (load_e_undirected_H) {
      const size_t count = e_undirected_H_data->count;
      Samplesi e_undirected_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(e_undirected_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        e_undirected_H(i, 0) = data_ptr[i];
      }
      mesh_samples["e_undirected_H"] = e_undirected_H;
    }
    if (load_f_left_H) {
      const size_t count = f_left_H_data->count;
      Samplesi f_left_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(f_left_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        f_left_H(i, 0) = data_ptr[i];
      }
      mesh_samples["f_left_H"] = f_left_H;
    }
    if (load_c_below_H) {
      const size_t count = c_below_H_data->count;
      Samplesi c_below_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(c_below_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        c_below_H(i, 0) = data_ptr[i];
      }
      mesh_samples["c_below_H"] = c_below_H;
    }
    if (load_h_next_H) {
      const size_t count = h_next_H_data->count;
      Samplesi h_next_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_next_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_next_H(i, 0) = data_ptr[i];
      }
      mesh_samples["h_next_H"] = h_next_H;
    }
    if (load_h_twin_H) {
      const size_t count = h_twin_H_data->count;
      Samplesi h_twin_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_twin_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_twin_H(i, 0) = data_ptr[i];
      }
      mesh_samples["h_twin_H"] = h_twin_H;
    }
    if (load_h_flip_H) {
      const size_t count = h_flip_H_data->count;
      Samplesi h_flip_H(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(h_flip_H_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        h_flip_H(i, 0) = data_ptr[i];
      }
      mesh_samples["h_flip_H"] = h_flip_H;
    }

    ///////////////////////////////////////////
    // Darts
    ///////////////////////////////////////////
    if (load_v_in_D) {
      const size_t count = v_in_D_data->count;
      Samplesi v_in_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(v_in_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        v_in_D(i, 0) = data_ptr[i];
      }
      mesh_samples["v_in_D"] = v_in_D;
    }
    if (load_e_in_D) {
      const size_t count = e_in_D_data->count;
      Samplesi e_in_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(e_in_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        e_in_D(i, 0) = data_ptr[i];
      }
      mesh_samples["e_in_D"] = e_in_D;
    }
    if (load_f_in_D) {
      const size_t count = f_in_D_data->count;
      Samplesi f_in_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(f_in_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        f_in_D(i, 0) = data_ptr[i];
      }
      mesh_samples["f_in_D"] = f_in_D;
    }
    if (load_c_in_D) {
      const size_t count = c_in_D_data->count;
      Samplesi c_in_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(c_in_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        c_in_D(i, 0) = data_ptr[i];
      }
      mesh_samples["c_in_D"] = c_in_D;
    }
    if (load_d_diff0_D) {
      const size_t count = d_diff0_D_data->count;
      Samplesi d_diff0_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_diff0_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_diff0_D(i, 0) = data_ptr[i];
      }
      mesh_samples["d_diff0_D"] = d_diff0_D;
    }
    if (load_d_diff1_D) {
      const size_t count = d_diff1_D_data->count;
      Samplesi d_diff1_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_diff1_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_diff1_D(i, 0) = data_ptr[i];
      }
      mesh_samples["d_diff1_D"] = d_diff1_D;
    }
    if (load_d_diff2_D) {
      const size_t count = d_diff2_D_data->count;
      Samplesi d_diff2_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_diff2_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_diff2_D(i, 0) = data_ptr[i];
      }
      mesh_samples["d_diff2_D"] = d_diff2_D;
    }
    if (load_d_diff3_D) {
      const size_t count = d_diff3_D_data->count;
      Samplesi d_diff3_D(count, 1);
      const int *data_ptr =
          reinterpret_cast<const int *>(d_diff3_D_data->buffer.get());
      for (size_t i = 0; i < count; ++i) {
        d_diff3_D(i, 0) = data_ptr[i];
      }
      mesh_samples["d_diff3_D"] = d_diff3_D;
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }
  return mesh_samples;
}

void write_mesh_samples_to_ply(const MeshSamples32 &mesh_samples,
                               const std::string &ply_path,
                               const bool use_binary) {

  using Samplesi = SamplesTypeDimTemplate<std::int32_t, 1>;
  using Samples2i = SamplesTypeDimTemplate<std::int32_t, 2>;
  using Samples3i = SamplesTypeDimTemplate<std::int32_t, 3>;
  using Samples4i = SamplesTypeDimTemplate<std::int32_t, 4>;

  std::vector<std::uint8_t> rgba_u8;

  std::filebuf fb;
  fb.open(ply_path,
          use_binary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + ply_path);

  tinyply::PlyFile mesh_file;

  const Samples3d *xyz_coord_V = nullptr;
  const Samplesi *h_out_V = nullptr;
  const Samplesi *d_through_V = nullptr;
  const Samples4i *rgba_V = nullptr;

  const Samples2i *V_cycle_E = nullptr;   // **
  const Samplesi *h_directed_E = nullptr; // **
  const Samplesi *d_through_E = nullptr;  // **

  const Samples3i *V_cycle_F = nullptr; // **
  const Samplesi *h_right_F = nullptr;
  const Samplesi *d_through_F = nullptr; // **

  const Samples4i *V_cycle_C = nullptr;  // **
  const Samplesi *h_above_C = nullptr;   // **
  const Samplesi *d_through_C = nullptr; // **

  const Samplesi *h_negative_B = nullptr;
  const Samplesi *d_through_B = nullptr; // **

  const Samplesi *v_origin_H = nullptr;
  const Samplesi *e_undirected_H = nullptr; // **
  const Samplesi *f_left_H = nullptr;
  const Samplesi *c_below_H = nullptr; // ***
  const Samplesi *h_next_H = nullptr;
  const Samplesi *h_twin_H = nullptr;
  const Samplesi *h_flip_H = nullptr; // ***

  const Samplesi *v_in_D = nullptr;    //**
  const Samplesi *e_in_D = nullptr;    //**
  const Samplesi *f_in_D = nullptr;    //**
  const Samplesi *c_in_D = nullptr;    //**
  const Samplesi *d_diff0_D = nullptr; //**
  const Samplesi *d_diff1_D = nullptr; //**
  const Samplesi *d_diff2_D = nullptr; //**
  const Samplesi *d_diff3_D = nullptr; //**

  // Retrieve samples from the map if they exist
  ////////////////////////////////////////////
  // Vertices
  if (auto it = mesh_samples.find("xyz_coord_V"); it != mesh_samples.end()) {
    xyz_coord_V = std::get_if<Samples3d>(&it->second);
    if (!xyz_coord_V) {
      throw std::runtime_error(
          R"(mesh_samples["xyz_coord_V"] exists but is not Samples3d)");
    }
    if (xyz_coord_V->cols() != 3) {
      throw std::runtime_error("xyz_coord_V must be (#V x 3)");
    }
    mesh_file.add_properties_to_element(
        "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
        static_cast<uint32_t>(xyz_coord_V->rows()),
        reinterpret_cast<uint8_t *>(const_cast<double *>(xyz_coord_V->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("h_out_V"); it != mesh_samples.end()) {
    h_out_V = std::get_if<Samplesi>(&it->second);
    if (!h_out_V) {
      throw std::runtime_error(
          R"(mesh_samples["h_out_V"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "vertex", {"h_out"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_out_V->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_out_V->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_through_V"); it != mesh_samples.end()) {
    d_through_V = std::get_if<Samplesi>(&it->second);
    if (!d_through_V) {
      throw std::runtime_error(
          R"(mesh_samples["d_through_V"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "vertex", {"d_through"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_through_V->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_through_V->data())),
        tinyply::Type::INVALID, 0);
  }
  // if (auto it = mesh_samples.find("rgba_V"); it != mesh_samples.end()) {
  //   rgba_V = std::get_if<Samples4i>(&it->second);
  //   if (!rgba_V) {
  //     throw std::runtime_error(
  //         R"(mesh_samples["rgba_V"] exists but is not Samples4i)");
  //   }
  //   if (rgba_V->cols() != 4) {
  //     throw std::runtime_error("rgba_V must be (#V x 4)");
  //   }
  //   mesh_file.add_properties_to_element(
  //       "vertex", {"red", "green", "blue", "alpha"}, tinyply::Type::INT32,
  //       static_cast<uint32_t>(rgba_V->rows()),
  //       reinterpret_cast<uint8_t *>(const_cast<int *>(rgba_V->data())),
  //       tinyply::Type::INVALID, 0);
  // }
  if (auto it = mesh_samples.find("rgba_V"); it != mesh_samples.end()) {
    rgba_V = std::get_if<Samples4i>(&it->second);
    if (!rgba_V)
      throw std::runtime_error(
          R"(mesh_samples["rgba_V"] exists but is not Samples4i)");
    if (rgba_V->cols() != 4)
      throw std::runtime_error("rgba_V must be (#V x 4)");

    rgba_u8.resize(static_cast<size_t>(rgba_V->rows()) * 4);
    for (Eigen::Index i = 0; i < rgba_V->rows(); ++i) {
      for (int j = 0; j < 4; ++j) {
        int x = (*rgba_V)(i, j);
        if (x < 0)
          x = 0;
        if (x > 255)
          x = 255;
        rgba_u8[static_cast<size_t>(i) * 4 + j] = static_cast<std::uint8_t>(x);
      }
    }

    mesh_file.add_properties_to_element(
        "vertex", {"red", "green", "blue", "alpha"}, tinyply::Type::UINT8,
        static_cast<uint32_t>(rgba_V->rows()),
        reinterpret_cast<uint8_t *>(rgba_u8.data()), tinyply::Type::INVALID, 0);
  }
  ////////////////////////////////////////////
  // Edges
  if (auto it = mesh_samples.find("V_cycle_E"); it != mesh_samples.end()) {
    V_cycle_E = std::get_if<Samples2i>(&it->second);
    if (!V_cycle_E) {
      throw std::runtime_error(
          R"(mesh_samples["V_cycle_E"] exists but is not Samples2i)");
    }
    if (V_cycle_E->cols() != 2) {
      throw std::runtime_error("V_cycle_E must be (#E x 2)");
    }
    mesh_file.add_properties_to_element(
        "edge", {"vertex_indices"}, tinyply::Type::INT32,
        static_cast<uint32_t>(V_cycle_E->rows()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(V_cycle_E->data())),
        tinyply::Type::UINT8, // list-count type in the file
        2                     // number of ints per edge
    );
  }
  if (auto it = mesh_samples.find("h_directed_E"); it != mesh_samples.end()) {
    h_directed_E = std::get_if<Samplesi>(&it->second);
    if (!h_directed_E) {
      throw std::runtime_error(
          R"(mesh_samples["h_directed_E"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "edge", {"h_directed"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_directed_E->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_directed_E->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_through_E"); it != mesh_samples.end()) {
    d_through_E = std::get_if<Samplesi>(&it->second);
    if (!d_through_E) {
      throw std::runtime_error(
          R"(mesh_samples["d_through_E"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "edge", {"d_through"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_through_E->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_through_E->data())),
        tinyply::Type::INVALID, 0);
  }
  ////////////////////////////////////////////
  // Faces
  if (auto it = mesh_samples.find("V_cycle_F"); it != mesh_samples.end()) {
    V_cycle_F = std::get_if<Samples3i>(&it->second);
    if (!V_cycle_F) {
      throw std::runtime_error(
          R"(mesh_samples["V_cycle_F"] exists but is not Samples3i)");
    }
    if (V_cycle_F->cols() != 3) {
      throw std::runtime_error("V_cycle_F must be (#F x 3)");
    }
    mesh_file.add_properties_to_element(
        "face", {"vertex_indices"}, tinyply::Type::INT32,
        static_cast<uint32_t>(V_cycle_F->rows()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(V_cycle_F->data())),
        tinyply::Type::UINT8, // list-count type in the file
        3                     // number of ints per face
    );
  }
  if (auto it = mesh_samples.find("h_right_F"); it != mesh_samples.end()) {
    h_right_F = std::get_if<Samplesi>(&it->second);
    if (!h_right_F) {
      throw std::runtime_error(
          R"(mesh_samples["h_right_F"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "face", {"h_right"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_right_F->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_right_F->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_through_F"); it != mesh_samples.end()) {
    d_through_F = std::get_if<Samplesi>(&it->second);
    if (!d_through_F) {
      throw std::runtime_error(
          R"(mesh_samples["d_through_F"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "face", {"d_through"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_through_F->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_through_F->data())),
        tinyply::Type::INVALID, 0);
  }
  ////////////////////////////////////////////
  // Cells
  if (auto it = mesh_samples.find("V_cycle_C"); it != mesh_samples.end()) {
    V_cycle_C = std::get_if<Samples4i>(&it->second);
    if (!V_cycle_C) {
      throw std::runtime_error(
          R"(mesh_samples["V_cycle_C"] exists but is not Samples4i)");
    }
    if (V_cycle_C->cols() != 4) {
      throw std::runtime_error("V_cycle_C must be (#C x 4)");
    }
    mesh_file.add_properties_to_element(
        "cell", {"vertex_indices"}, tinyply::Type::INT32,
        static_cast<uint32_t>(V_cycle_C->rows()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(V_cycle_C->data())),
        tinyply::Type::UINT8, // list-count type in the file
        4                     // number of ints per cell
    );
  }
  if (auto it = mesh_samples.find("h_above_C"); it != mesh_samples.end()) {
    h_above_C = std::get_if<Samplesi>(&it->second);
    if (!h_above_C) {
      throw std::runtime_error(
          R"(mesh_samples["h_above_C"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "cell", {"h_above"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_above_C->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_above_C->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_through_C"); it != mesh_samples.end()) {
    d_through_C = std::get_if<Samplesi>(&it->second);
    if (!d_through_C) {
      throw std::runtime_error(
          R"(mesh_samples["d_through_C"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "cell", {"d_through"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_through_C->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_through_C->data())),
        tinyply::Type::INVALID, 0);
  }
  ////////////////////////////////////////////
  // try { tripstrip = file.request_properties_from_element("tristrips", {
  // "vertex_indices" }, 0); } catch (const std::exception & e) { std::cerr <<
  // "tinyply exception: " << e.what() << std::endl; }

  // Boundaries
  if (auto it = mesh_samples.find("h_negative_B"); it != mesh_samples.end()) {
    h_negative_B = std::get_if<Samplesi>(&it->second);
    if (!h_negative_B) {
      throw std::runtime_error(
          R"(mesh_samples["h_negative_B"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "boundary", {"h_negative"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_negative_B->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_negative_B->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_through_B"); it != mesh_samples.end()) {
    d_through_B = std::get_if<Samplesi>(&it->second);
    if (!d_through_B) {
      throw std::runtime_error(
          R"(mesh_samples["d_through_B"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "boundary", {"d_through"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_through_B->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_through_B->data())),
        tinyply::Type::INVALID, 0);
  }
  ////////////////////////////////////////////
  // Half-edges
  if (auto it = mesh_samples.find("v_origin_H"); it != mesh_samples.end()) {
    v_origin_H = std::get_if<Samplesi>(&it->second);
    if (!v_origin_H) {
      throw std::runtime_error(
          R"(mesh_samples["v_origin_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"v_origin"}, tinyply::Type::INT32,
        static_cast<uint32_t>(v_origin_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(v_origin_H->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("e_undirected_H"); it != mesh_samples.end()) {
    e_undirected_H = std::get_if<Samplesi>(&it->second);
    if (!e_undirected_H) {
      throw std::runtime_error(
          R"(mesh_samples["e_undirected_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"e_undirected"}, tinyply::Type::INT32,
        static_cast<uint32_t>(e_undirected_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(e_undirected_H->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("f_left_H"); it != mesh_samples.end()) {
    f_left_H = std::get_if<Samplesi>(&it->second);
    if (!f_left_H) {
      throw std::runtime_error(
          R"(mesh_samples["f_left_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"f_left"}, tinyply::Type::INT32,
        static_cast<uint32_t>(f_left_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(f_left_H->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("c_below_H"); it != mesh_samples.end()) {
    c_below_H = std::get_if<Samplesi>(&it->second);
    if (!c_below_H) {
      throw std::runtime_error(
          R"(mesh_samples["c_below_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"c_below"}, tinyply::Type::INT32,
        static_cast<uint32_t>(c_below_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(c_below_H->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("h_next_H"); it != mesh_samples.end()) {
    h_next_H = std::get_if<Samplesi>(&it->second);
    if (!h_next_H) {
      throw std::runtime_error(
          R"(mesh_samples["h_next_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"h_next"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_next_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_next_H->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("h_twin_H"); it != mesh_samples.end()) {
    h_twin_H = std::get_if<Samplesi>(&it->second);
    if (!h_twin_H) {
      throw std::runtime_error(
          R"(mesh_samples["h_twin_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"h_twin"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_twin_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_twin_H->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("h_flip_H"); it != mesh_samples.end()) {
    h_flip_H = std::get_if<Samplesi>(&it->second);
    if (!h_flip_H) {
      throw std::runtime_error(
          R"(mesh_samples["h_flip_H"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "half_edge", {"h_flip"}, tinyply::Type::INT32,
        static_cast<uint32_t>(h_flip_H->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(h_flip_H->data())),
        tinyply::Type::INVALID, 0);
  }
  ////////////////////////////////////////////
  // Darts
  if (auto it = mesh_samples.find("v_in_D"); it != mesh_samples.end()) {
    v_in_D = std::get_if<Samplesi>(&it->second);
    if (!v_in_D) {
      throw std::runtime_error(
          R"(mesh_samples["v_in_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"v_in"}, tinyply::Type::INT32,
        static_cast<uint32_t>(v_in_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(v_in_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("e_in_D"); it != mesh_samples.end()) {
    e_in_D = std::get_if<Samplesi>(&it->second);
    if (!e_in_D) {
      throw std::runtime_error(
          R"(mesh_samples["e_in_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"e_in"}, tinyply::Type::INT32,
        static_cast<uint32_t>(e_in_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(e_in_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("f_in_D"); it != mesh_samples.end()) {
    f_in_D = std::get_if<Samplesi>(&it->second);
    if (!f_in_D) {
      throw std::runtime_error(
          R"(mesh_samples["f_in_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"f_in"}, tinyply::Type::INT32,
        static_cast<uint32_t>(f_in_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(f_in_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("c_in_D"); it != mesh_samples.end()) {
    c_in_D = std::get_if<Samplesi>(&it->second);
    if (!c_in_D) {
      throw std::runtime_error(
          R"(mesh_samples["c_in_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"c_in"}, tinyply::Type::INT32,
        static_cast<uint32_t>(c_in_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(c_in_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_diff0_D"); it != mesh_samples.end()) {
    d_diff0_D = std::get_if<Samplesi>(&it->second);
    if (!d_diff0_D) {
      throw std::runtime_error(
          R"(mesh_samples["d_diff0_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"d_diff0"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_diff0_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_diff0_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_diff1_D"); it != mesh_samples.end()) {
    d_diff1_D = std::get_if<Samplesi>(&it->second);
    if (!d_diff1_D) {
      throw std::runtime_error(
          R"(mesh_samples["d_diff1_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"d_diff1"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_diff1_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_diff1_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_diff2_D"); it != mesh_samples.end()) {
    d_diff2_D = std::get_if<Samplesi>(&it->second);
    if (!d_diff2_D) {
      throw std::runtime_error(
          R"(mesh_samples["d_diff2_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"d_diff2"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_diff2_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_diff2_D->data())),
        tinyply::Type::INVALID, 0);
  }
  if (auto it = mesh_samples.find("d_diff3_D"); it != mesh_samples.end()) {
    d_diff3_D = std::get_if<Samplesi>(&it->second);
    if (!d_diff3_D) {
      throw std::runtime_error(
          R"(mesh_samples["d_diff3_D"] exists but is not Samplesi)");
    }
    mesh_file.add_properties_to_element(
        "dart", {"d_diff3"}, tinyply::Type::INT32,
        static_cast<uint32_t>(d_diff3_D->size()),
        reinterpret_cast<uint8_t *>(const_cast<int *>(d_diff3_D->data())),
        tinyply::Type::INVALID, 0);
  }
  mesh_file.get_comments().push_back("MathUtils ply");

  // Write an ply file
  mesh_file.write(outstream, use_binary);
}
} // namespace io
} // namespace mesh
} // namespace mathutils