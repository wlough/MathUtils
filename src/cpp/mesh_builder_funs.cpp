#pragma once

/**
 * @file mesh_builder_funs.cpp
 */

#include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/matrix.hpp"
#include "mathutils/mesh/mesh_common.hpp"

namespace mathutils {
namespace mesh {
MeshSamples build_icosohedron() {
  printf("MatrixMesh::from_icosohedron\n");
  MeshSamples ms;
  double phi = (1.0 + sqrt(5.0)) * 0.5; // golden ratio
  double a = 1.0;
  double b = 1.0 / phi;

  int num_vertices = 12;
  int num_faces = 20;
  Matrix<Real> X_ambient_V(num_vertices, 3);
  Matrix<Index> V_cycle_F(num_faces, 3);
  Matrix<Index> V_cycle_E(num_faces, 2);

  X_ambient_V.set_row(0, {0.0, b, -a});
  X_ambient_V.set_row(1, {b, a, 0.0});
  X_ambient_V.set_row(2, {-b, a, 0.0});
  X_ambient_V.set_row(3, {0.0, b, a});
  X_ambient_V.set_row(4, {0.0, -b, a});
  X_ambient_V.set_row(5, {-a, 0.0, b});
  X_ambient_V.set_row(6, {0.0, -b, -a});
  X_ambient_V.set_row(7, {a, 0.0, -b});
  X_ambient_V.set_row(8, {a, 0.0, b});
  X_ambient_V.set_row(9, {-a, 0.0, -b});
  X_ambient_V.set_row(10, {b, -a, 0.0});
  X_ambient_V.set_row(11, {-b, -a, 0.0});

  double rad = std::sqrt(a * a + b * b);
  X_ambient_V *= 1.0 / rad;

  V_cycle_F.set_row(0, {2, 1, 0});
  V_cycle_F.set_row(1, {1, 2, 3});
  V_cycle_F.set_row(2, {5, 4, 3});
  V_cycle_F.set_row(3, {4, 8, 3});
  V_cycle_F.set_row(4, {7, 6, 0});
  V_cycle_F.set_row(5, {6, 9, 0});
  V_cycle_F.set_row(6, {11, 10, 4});
  V_cycle_F.set_row(7, {10, 11, 6});
  V_cycle_F.set_row(8, {9, 5, 2});
  V_cycle_F.set_row(9, {5, 9, 11});
  V_cycle_F.set_row(10, {8, 7, 1});
  V_cycle_F.set_row(11, {7, 8, 10});
  V_cycle_F.set_row(12, {2, 5, 3});
  V_cycle_F.set_row(13, {8, 1, 3});
  V_cycle_F.set_row(14, {9, 2, 0});
  V_cycle_F.set_row(15, {1, 7, 0});
  V_cycle_F.set_row(16, {11, 9, 6});
  V_cycle_F.set_row(17, {7, 10, 6});
  V_cycle_F.set_row(18, {5, 11, 4});
  V_cycle_F.set_row(19, {10, 8, 4});

  V_cycle_E.set_row(0, {2, 1, 0});

  ms["X_ambient_V"] = X_ambient_V;
  ms["V_cycle_F"] = V_cycle_F;
  return ms;
}

void refine_vertex_face_samples(MeshSamples &ms) {
  // printf("MatrixMesh::refine_icososphere\n");

  // auto vertex_pair_key = [](Index v0, Index v1) -> long long {
  //   return static_cast<long long>(std::min(v0, v1)) *
  //              static_cast<long long>(1000000) +
  //          static_cast<long long>(std::max(v0, v1));
  // };
  auto vertex_pair_key = [](Index v0, Index v1) -> std::uint64_t {
    std::uint32_t a = static_cast<std::uint32_t>(std::min(v0, v1));
    std::uint32_t b = static_cast<std::uint32_t>(std::max(v0, v1));
    return (std::uint64_t(a) << 32) | std::uint64_t(b);
  };
  // auto vertex_pair_key = [](Index v0, Index v1) -> long long {
  //   return static_cast<long long>(std::min(v0, v1)) *
  //              static_cast<long long>(1000000) +
  //          static_cast<long long>(std::max(v0, v1));
  // };
  // Matrix<Real> X_ambient_V;
  // SamplesIndex V_cycle_F0;

  SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
  SamplesIndex &V_cycle_F0 = std::get<SamplesIndex>(ms.at("V_cycle_F"));

  Index num_faces0 = V_cycle_F0.rows();
  Index num_faces = 4 * num_faces0;
  Index num_vertices0 = X_ambient_V.rows();

  SamplesIndex V_cycle_F(num_faces, 3);

  std::vector<Matrix<Real>> vecV;
  vecV.reserve(num_vertices0);
  for (Index v = 0; v < num_vertices0; v++) {
    vecV.push_back(X_ambient_V.row_copy(v));
  }

  std::unordered_map<std::uint64_t, Index> v_midpt_vv;
  Index v_count = num_vertices0;
  Index f_count = 0;
  for (Index f = 0; f < num_faces0; f++) {
    // printf("  f=%d\n", f);
    Index v0 = V_cycle_F0(f, 0);
    Index v1 = V_cycle_F0(f, 1);
    Index v2 = V_cycle_F0(f, 2);
    // printf("  v0,v1,v2=%d,%d,%d\n", v0, v1, v2);
    long long key01 = vertex_pair_key(v0, v1);
    long long key12 = vertex_pair_key(v1, v2);
    long long key20 = vertex_pair_key(v2, v0);
    // std::cout << "  key01, key12, key20=" << key01 << "," << key12 << ","
    //           << key20 << std::endl;

    Index v01 =
        (v_midpt_vv.count(key01) == 1) ? v_midpt_vv[key01] : InvalidIndex;
    Index v12 =
        (v_midpt_vv.count(key12) == 1) ? v_midpt_vv[key12] : InvalidIndex;
    Index v20 =
        (v_midpt_vv.count(key20) == 1) ? v_midpt_vv[key20] : InvalidIndex;
    // printf("  v01,v12,v20=%d,%d,%d\n", v01, v12, v20);
    if (v01 == InvalidIndex) {
      v01 = v_count++;

      Matrix<Real> xyz01 = (vecV[v0] + vecV[v1]) / 2.0;
      xyz01 *= 1 / xyz01.norm();
      // vecV[v01] = xyz01;
      vecV.push_back(xyz01);
      v_midpt_vv[key01] = v01;
    }
    if (v12 == InvalidIndex) {
      v12 = v_count++;
      Matrix<Real> xyz12 = (vecV[v1] + vecV[v2]) / 2.0;
      xyz12 *= 1 / xyz12.norm();
      // vecV[v12] = xyz12;
      vecV.push_back(xyz12);
      v_midpt_vv[key12] = v12;
    }
    if (v20 == InvalidIndex) {
      v20 = v_count++;
      Matrix<Real> xyz20 = (vecV[v2] + vecV[v0]) / 2.0;
      xyz20 *= 1 / xyz20.norm();
      // vecV[v20] = xyz20;
      vecV.push_back(xyz20);
      v_midpt_vv[key20] = v20;
    }

    // printf("  v01,v12,v20=%d,%d,%d\n", v01, v12, v20);

    V_cycle_F.set_row(f_count++, {v0, v01, v20});
    V_cycle_F.set_row(f_count++, {v01, v1, v12});
    V_cycle_F.set_row(f_count++, {v20, v12, v2});
    V_cycle_F.set_row(f_count++, {v01, v12, v20});
  }

  X_ambient_V.conservativeResize(vecV.size(), 3);
  for (Index v = 0; v < vecV.size(); v++) {
    vecV[v] /= vecV[v].norm();
    X_ambient_V.set_row(v, {vecV[v][0], vecV[v][1], vecV[v][2]});
  }

  ms["X_ambienoutV"] = X_ambient_V;
  ms["V_cycle_F"] = V_cycle_F;
}

MeshSamples build_icososphere(size_t num_refinements) {
  MeshSamples ms = build_icosohedron();

  for (size_t refinement; refinement < num_refinements; ++refinement) {
    refine_vertex_face_samples(ms);
    SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
    size_t num_vertices = X_ambient_V.rows();
  }

  return ms;
}
} // namespace mesh
} // namespace mathutils
