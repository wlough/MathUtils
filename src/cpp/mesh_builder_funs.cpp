/**
 * @file mesh_builder_funs.cpp
 */

#include "mathutils/mesh/mesh_builder_funs.hpp"
#include "mathutils/matrix.hpp"
#include "mathutils/mesh/mesh_common.hpp"

namespace mathutils {
namespace mesh {
MeshSamples build_icosohedron_samples() {
  printf("MatrixMesh::from_icosohedron\n");
  MeshSamples ms;
  double phi = (1.0 + sqrt(5.0)) * 0.5; // golden ratio 1.61803...
  double a = 1.0;
  double b = 1.0 / phi;

  int num_vertices = 12;
  int num_edges = 30;
  int num_faces = 20;
  Matrix<Real> X_ambient_V(num_vertices, 3);
  Matrix<Index> V_cycle_E(num_edges, 2);
  Matrix<Index> V_cycle_F(num_faces, 3);

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

  // {2, 1, 0} -> {1, 2}, {0, 1}, {0, 2}
  // {1, 2, 3} -> {1, 2}, {2, 3}, {1, 3}
  // {5, 4, 3} -> {4, 5}, {3, 4}, {3, 5}
  // {4, 8, 3} -> {4, 8}, {3, 8}, {3, 4}
  // {7, 6, 0} -> {6, 7}, {0, 6}, {0, 7}
  // {6, 9, 0} -> {6, 9}, {0, 9}, {0, 6}
  // {11, 10, 4} -> {10, 11}, {4, 10}, {4, 11}
  // {10, 11, 6} -> {10, 11}, {6, 11}, {6, 10}
  // {9, 5, 2} -> {5, 9}, {2, 5}, {2, 9}
  // {5, 9, 11} -> {5, 9}, {9, 11}, {5, 11}
  // {8, 7, 1} -> {7, 8}, {1, 7}, {1, 8}
  // {7, 8, 10} -> {7, 8}, {8, 10}, {7, 10}
  // {2, 5, 3} -> {2, 5}, {3, 5}, {2, 3}
  // {8, 1, 3} -> {1, 8}, {1, 3}, {3, 8}
  // {9, 2, 0} -> {2, 9}, {0, 2}, {0, 9}
  // {1, 7, 0} -> {1, 7}, {0, 7}, {0, 1}
  // {11, 9, 6} -> {9, 11}, {6, 9}, {6, 11}
  // {7, 10, 6} -> {7, 10}, {6, 10}, {6, 7}
  // {5, 11, 4} -> {5, 11}, {4, 11}, {4, 5}
  // {10, 8, 4} -> {8, 10}, {4, 8}, {4, 10}

  // {0, 1}, {0, 2}, {1, 2}
  // {1, 2}, {1, 3}, {2, 3}
  // {3, 4}, {3, 5}, {4, 5}
  // {3, 4}, {3, 8}, {4, 8}
  // {0, 6}, {0, 7}, {6, 7}
  // {0, 6}, {0, 9}, {6, 9}
  // {4, 10}, {4, 11}, {10, 11}
  // {6, 10}, {6, 11}, {10, 11}
  // {2, 5}, {2, 9}, {5, 9}
  // {5, 9}, {5, 11}, {9, 11}
  // {1, 7}, {1, 8}, {7, 8}
  // {7, 8}, {7, 10}, {8, 10}
  // {2, 3}, {2, 5}, {3, 5}
  // {1, 3}, {1, 8}, {3, 8}
  // {0, 2}, {0, 9}, {2, 9}
  // {0, 1}, {0, 7}, {1, 7}
  // {6, 9}, {6, 11}, {9, 11}
  // {6, 7}, {6, 10}, {7, 10}
  // {4, 5}, {4, 11}, {5, 11}
  // {4, 8}, {4, 10}, {8, 10}

  V_cycle_E.set_row(0, {0, 1});
  V_cycle_E.set_row(1, {0, 2});
  V_cycle_E.set_row(2, {0, 6});
  V_cycle_E.set_row(3, {0, 7});
  V_cycle_E.set_row(4, {0, 9});
  V_cycle_E.set_row(5, {1, 2});
  V_cycle_E.set_row(6, {1, 3});
  V_cycle_E.set_row(7, {1, 7});
  V_cycle_E.set_row(8, {1, 8});
  V_cycle_E.set_row(9, {2, 3});
  V_cycle_E.set_row(10, {2, 5});
  V_cycle_E.set_row(11, {2, 9});
  V_cycle_E.set_row(12, {3, 4});
  V_cycle_E.set_row(13, {3, 5});
  V_cycle_E.set_row(14, {3, 8});
  V_cycle_E.set_row(15, {4, 5});
  V_cycle_E.set_row(16, {4, 8});
  V_cycle_E.set_row(17, {4, 10});
  V_cycle_E.set_row(18, {4, 11});
  V_cycle_E.set_row(19, {5, 9});
  V_cycle_E.set_row(20, {5, 11});
  V_cycle_E.set_row(21, {6, 7});
  V_cycle_E.set_row(22, {6, 9});
  V_cycle_E.set_row(23, {6, 10});
  V_cycle_E.set_row(24, {6, 11});
  V_cycle_E.set_row(25, {7, 8});
  V_cycle_E.set_row(26, {7, 10});
  V_cycle_E.set_row(27, {8, 10});
  V_cycle_E.set_row(28, {9, 11});
  V_cycle_E.set_row(29, {10, 11});

  ms["X_ambient_V"] = X_ambient_V;
  ms["V_cycle_E"] = V_cycle_E;
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

  SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
  SamplesIndex &V_cycle_F0 = std::get<SamplesIndex>(ms.at("V_cycle_F"));

  Index num_faces0 = V_cycle_F0.rows();
  Index num_faces = 4 * num_faces0;
  Index num_vertices0 = X_ambient_V.rows();

  SamplesIndex V_cycle_F(num_faces, 3);

  std::vector<SamplesReal> newV;
  newV.reserve(num_vertices0 +
               num_faces0); // num_vertices0 + num_faces0 - 2 for sphere

  std::unordered_map<std::uint64_t, Index> v_midpt_vv;
  Index v_count = num_vertices0;
  Index f_count = 0;

  printf("for (Index f = 0; f < num_faces0; f++)");
  for (Index f = 0; f < num_faces0; f++) {
    Index v0 = V_cycle_F0(f, 0);
    Index v1 = V_cycle_F0(f, 1);
    Index v2 = V_cycle_F0(f, 2);
    long long key01 = vertex_pair_key(v0, v1);
    long long key12 = vertex_pair_key(v1, v2);
    long long key20 = vertex_pair_key(v2, v0);

    Index v01 =
        (v_midpt_vv.count(key01) == 1) ? v_midpt_vv[key01] : InvalidIndex;
    Index v12 =
        (v_midpt_vv.count(key12) == 1) ? v_midpt_vv[key12] : InvalidIndex;
    Index v20 =
        (v_midpt_vv.count(key20) == 1) ? v_midpt_vv[key20] : InvalidIndex;

    if (v01 == InvalidIndex) {
      v01 = v_count++;
      SamplesReal xyz01 =
          (X_ambient_V.row_copy(v0) + X_ambient_V.row_copy(v1)) / 2.0;
      newV.push_back(xyz01);
      v_midpt_vv[key01] = v01;
    }
    if (v12 == InvalidIndex) {
      v12 = v_count++;
      SamplesReal xyz12 =
          (X_ambient_V.row_copy(v1) + X_ambient_V.row_copy(v2)) / 2.0;
      newV.push_back(xyz12);
      v_midpt_vv[key12] = v12;
    }
    if (v20 == InvalidIndex) {
      v20 = v_count++;
      SamplesReal xyz20 =
          (X_ambient_V.row_copy(v2) + X_ambient_V.row_copy(v0)) / 2.0;
      newV.push_back(xyz20);
      v_midpt_vv[key20] = v20;
    }

    V_cycle_F.set_row(f_count++, {v0, v01, v20});
    V_cycle_F.set_row(f_count++, {v01, v1, v12});
    V_cycle_F.set_row(f_count++, {v20, v12, v2});
    V_cycle_F.set_row(f_count++, {v01, v12, v20});
  }

  X_ambient_V.conservativeResize(num_vertices0 + newV.size(), 3);
  printf("for (Index v = 0; v < newV.size(); v++)");
  for (Index v = 0; v < newV.size(); v++) {
    X_ambient_V.set_row(v + num_vertices0,
                        {newV[v][0], newV[v][1], newV[v][2]});
  }

  ms["X_ambientV"] = X_ambient_V;
  ms["V_cycle_F"] = V_cycle_F;
}

MeshSamples build_icososphere_samples(size_t num_refinements) {
  MeshSamples ms = build_icosohedron_samples();
  SamplesReal &X_ambient_V0 = std::get<SamplesReal>(ms.at("X_ambient_V"));
  Index num_vertices0 = X_ambient_V0.rows();
  for (size_t refinement; refinement < num_refinements; ++refinement) {
    refine_vertex_face_samples(ms);
    Index num_vertices = X_ambient_V0.rows();
    for (Index v{num_vertices0}; v < num_vertices; ++v) {
      SamplesReal &X_ambient_V = std::get<SamplesReal>(ms.at("X_ambient_V"));
      SamplesReal xyz = X_ambient_V.row_copy(v);
      xyz /= xyz.norm();
      X_ambient_V.set_row(v, {xyz[0], xyz[1], xyz[2]});
    }
    num_vertices0 = num_vertices;
  }

  return ms;
}
} // namespace mesh
} // namespace mathutils
