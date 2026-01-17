/**
 * @file mesh.cpp
 */
#include "mathutils/mesh/mesh_convert_funs.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include <array>
#include <cstddef>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mathutils {
namespace mesh {

int find_halfedge_index_of_twin(const Samples2i &H, const int &h) {
  auto v0 = H(h, 0);
  auto v1 = H(h, 1);
  for (int h_twin = 0; h_twin < H.rows(); ++h_twin) {
    if ((H(h_twin, 0) == v1) && (H(h_twin, 1) == v0)) {
      return h_twin; // Return the index of the twin edge
    }
  }
  return -1; // Return -1 if no twin edge is found
}

std::map<std::string, Samplesi>
tri_vertex_cycles_to_half_edge_samples(const Samples3i &V_cycle_F) {

  auto Nv = V_cycle_F.maxCoeff() + 1;
  auto Nf = V_cycle_F.rows();
  // num interior + num positive boundary half-edges
  auto Nh0 = 3 * Nf;
  Samples2i H0 = Samples2i(Nh0, 2);
  // half-edge samples
  // h_out=Nh0 if not assigned
  // h_twin=-1 if not assigned
  Samplesi h_out_V = Samplesi::Constant(Nv, Nh0);
  Samplesi v_origin_H = Samplesi(Nh0);
  Samplesi h_next_H = Samplesi(Nh0);
  Samplesi h_twin_H = Samplesi::Constant(Nh0, -1);
  Samplesi f_left_H = Samplesi(Nh0);
  Samplesi h_right_F = Samplesi(Nf);
  Samplesi h_negative_B;
  // assign h_out for vertices to be minimum of outgoing half-edge indices
  // assign v_origin/f_left/h_next for half-edges in H0
  // assign h_bound for faces
  for (int f = 0; f < Nf; ++f) {
    h_right_F[f] = 3 * f;
    for (int i = 0; i < 3; ++i) {
      int h = 3 * f + i;
      int h_next = 3 * f + (i + 1) % 3;
      int v0 = V_cycle_F(f, i);
      int v1 = V_cycle_F(f, (i + 1) % 3);
      H0.row(h) << v0, v1;
      v_origin_H[h] = v0;
      f_left_H[h] = f;
      h_next_H[h] = h_next;
      // assign h_out for vertices if not already assigned
      // reassign if h is smaller than current h_out_V[v0]
      if (h_out_V[v0] > h) {
        h_out_V[v0] = h;
      }
    }
  }
  // Temporary containers for indices of +/- boundary half-edge
  std::vector<int> H_boundary_plus;
  std::unordered_set<int> H_boundary_minus;
  // find positive boundary half-edges
  // assign h_twin for interior half-edges
  for (int h = 0; h < H0.rows(); ++h) {
    // if h_twin_H[h] is already assigned, skip
    if (h_twin_H[h] != -1) {
      continue;
    }
    int h_twin = find_halfedge_index_of_twin(H0, h);
    if (h_twin == -1) {
      H_boundary_plus.push_back(h);
    } else {
      h_twin_H[h] = h_twin;
      h_twin_H[h_twin] = h;
    }
  }
  int Nh1 = H_boundary_plus.size();
  int Nh = Nh0 + Nh1;
  v_origin_H.conservativeResize(Nh);
  h_next_H.conservativeResize(Nh);
  h_twin_H.conservativeResize(Nh);
  f_left_H.conservativeResize(Nh);
  // define negative boundary half-edges
  // assign v_origin for negative boundary half-edges
  // assign h_twin for boundary half-edges
  for (int i = 0; i < Nh1; ++i) {
    int h = H_boundary_plus[i];
    int h_twin = Nh0 + i;
    // int v0 = H0(h, 0);
    int v1 = H0(h, 1);
    H_boundary_minus.insert(h_twin);
    v_origin_H[h_twin] = v1;
    h_twin_H[h] = h_twin;
    h_twin_H[h_twin] = h;
  }
  // enumerate boundaries b=0,1,...
  // assign h_right for boundaries
  // assign h_next for negative boundary half-edges
  // set f_left=-(b+1) for half-edges in boundary b
  while (!H_boundary_minus.empty()) {
    int b = h_negative_B.size();
    int h_negative_b = *H_boundary_minus.begin();
    h_negative_B.conservativeResize(b + 1);
    h_negative_B[b] = h_negative_b; // Assign new value
    int h = h_negative_b;
    // follow prev cycle along boundary b until we get back to h=h_negative_b
    do {
      int h_prev = h_twin_H[h];
      // rotate cw around origin of h until we find h_prev in boundary b
      // erase h from H_boundary_minus
      while (H_boundary_minus.find(h_prev) == H_boundary_minus.end()) {
        h_prev = h_twin_H[h_next_H[h_prev]];
      }
      h_next_H[h_prev] = h;
      h = h_prev;
      H_boundary_minus.erase(h);
      f_left_H[h] = -(b + 1);
    } while (h != h_negative_b);
  }
  std::map<std::string, Samplesi> halfedge_data_map;
  halfedge_data_map["h_out_V"] = h_out_V;
  halfedge_data_map["v_origin_H"] = v_origin_H;
  halfedge_data_map["h_next_H"] = h_next_H;
  halfedge_data_map["h_twin_H"] = h_twin_H;
  halfedge_data_map["f_left_H"] = f_left_H;
  halfedge_data_map["h_right_F"] = h_right_F;
  halfedge_data_map["h_negative_B"] = h_negative_B;
  return halfedge_data_map;
}

} // namespace mesh
} // namespace mathutils
