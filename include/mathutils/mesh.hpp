#pragma once

/**
 * @file mesh.hpp
 * @brief Mesh tools
 */

#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include "mathutils/simple_generator.hpp"
#include "simple_generator.hpp"


/////////////////////////////////////
/////////////////////////////////////
// Mesh utilities //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace mesh {
} // namespace mesh




/**
 * @brief Column vector of integers.
 */
using Samplesi = Eigen::VectorXi;
/**
 * @brief N-by-2 matrix of ints
 */
using Samples2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;
/**
 * @brief N-by-3 matrix of ints
 */
using Samples3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;
/**
 * @brief N-by-3 matrix of doubles. Represents 3D spatial coordinates of a
 * vertices in a vertex set.
 */
using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using Samples2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;



struct VertexSet {
  Samples3d coords_V;
};


struct EdgeSet {
  Samplesi V_cycle_E;

};



// half-verts
struct HalfVertData {
  Eigen::Matrix<double, Eigen::Dynamic, 3> xyz_coord_V;   //
  Eigen::Matrix<int, Eigen::Dynamic, 2> V_cycle_E;     //
                            //
  Eigen::VectorXi h_out_V;        //
  Eigen::VectorXi h_directed_E;   //
  Eigen::VectorXi h_negative_B;   //
                            //
  Eigen::VectorXi v_origin_H;     //
  Eigen::VectorXi e_undirected_H; //
                            //
  Eigen::VectorXi h_next_H;       //
  ////////////////////////////
};

class Vertex {
public:
  Eigen::Vector3d coords3d_;
  Eigen::Vector2d coords2d_;
};

class Edge {
public:
  Eigen::Vector3d coords3d_;
  Eigen::Vector2d coords2d_;
};




// half-edges
struct HalfEdgeData {
  Eigen::Matrix<double, Eigen::Dynamic, 3> xyz_coord_V;   //
  Eigen::Matrix<int, Eigen::Dynamic, 2> V_cycle_E;     //
  Eigen::Matrix<int, Eigen::Dynamic, 3> V_cycle_F;     //
                            //
  Eigen::VectorXi h_out_V;        //
  Eigen::VectorXi h_directed_E;   //
  Eigen::VectorXi h_right_F;      //
  Eigen::VectorXi h_negative_B;   //
                            //
  Eigen::VectorXi v_origin_H;     //
  Eigen::VectorXi e_undirected_H; //
  Eigen::VectorXi f_left_H;       //
                            //
  Eigen::VectorXi h_next_H;       //
  Eigen::VectorXi h_twin_H;       //
  ////////////////////////////
};

struct BoundaryLoop {
  HalfEdgeData he_data;
};



// half-faces
struct HalfFaceData {
  Eigen::Matrix<double, Eigen::Dynamic, 3> xyz_coord_V;   //
  Eigen::Matrix<int, Eigen::Dynamic, 2> V_cycle_E;     //
  Eigen::Matrix<int, Eigen::Dynamic, 3> V_cycle_F;     //
                            //
  Eigen::VectorXi h_out_V;        //
  Eigen::VectorXi h_directed_E;   //
  Eigen::VectorXi h_right_F;      //
  Eigen::VectorXi h_surf_C;      //
  Eigen::VectorXi h_negative_B;   //
                            //
  Eigen::VectorXi v_origin_H;     //
  Eigen::VectorXi e_undirected_H; //
  Eigen::VectorXi f_left_H;
  Eigen::VectorXi c_cobdry_H;       //
                            //
  Eigen::VectorXi h_next_H;       //
  Eigen::VectorXi h_twin_H;       //
  Eigen::VectorXi h_flip_H;       //
  ////////////////////////////
};


struct CombinatorialMap {
  Eigen::VectorXi h_out_V;        //
  Eigen::VectorXi h_directed_E;   //
  Eigen::VectorXi h_right_F;      //
  Eigen::VectorXi h_surf_C;      //
  Eigen::VectorXi h_negative_B;   //
                            //
  Eigen::VectorXi v_origin_H;     //
  Eigen::VectorXi e_undirected_H; //
  Eigen::VectorXi f_left_H;
  Eigen::VectorXi c_cobdry_H;       //
                            //
  Eigen::VectorXi h_next_H;       //
  Eigen::VectorXi h_twin_H;       //
  Eigen::VectorXi h_flip_H;       //
  ////////////////////////////
};




// mesh of control points for a surface
class DartMesh {
  // using mathutils::SimpleGenerator
  using GeneratorInt = mathutils::SimpleGenerator<int>;
public:
  GeneratorInt generate_V_cycle_f(int f) const;

  ////////////////////////////
  // Core data structure /////
  ////////////////////////////
  Samples3d xyz_coord_V_;
  Eigen::VectorXi h_out_V_;
  Eigen::VectorXi h_directed_E_;
  Eigen::VectorXi h_right_F_;
  Eigen::VectorXi h_above_C_;
  Eigen::VectorXi h_negative_B_;

  Eigen::VectorXi v_origin_H_;
  Eigen::VectorXi e_undirected_H_;
  Eigen::VectorXi f_left_H_;
  Eigen::VectorXi c_below_H_;

  Eigen::VectorXi h_next_H_;
  Eigen::VectorXi h_twin_H_;
  Eigen::VectorXi h_reverse_H_; // reflect?
  ////////////////////////////

  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////
  Vec3d xyz_coord_v(int v) const;
  int h_out_v(int v) const;
  int h_directed_e(int e) const;
  int h_right_f(int f) const;
  int h_above_c(int c) const;
  int h_negative_b(int b) const;

  int v_origin_h(int h) const;
  int e_undirected_h(int h) const;
  int f_left_h(int h) const;
  int c_below_h(int h) const;

  int h_next_h(int h) const; // (v', e', f, c)
  int h_twin_h(int h) const; // (v', e, f', c)
  int h_reverse_h(int h) const; // (v', e, f, c')

  // Derived combinatorial maps
  int h_beta0_h(int h) const; // (v', e, f, c)
  int h_beta1_h(int h) const {return ;} // (v, e', f, c)
  int h_beta2_h(int h) const; // (v, e, f', c)
  int h_beta3_h(int h) const; // (v, e, f, c')

  int h_in_v(int v) const;
  int v_head_h(int h) const;
  int h_prev_h(int h) const;
  int h_rotcw_h(int h) const;
  int h_rotccw_h(int h) const;
  int h_prev_h_by_rot(int h) const;

  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  bool some_negative_boundary_contains_h(int h) const;
  bool some_positive_boundary_contains_h(int h) const;
  bool some_boundary_contains_h(int h) const;
  bool some_boundary_contains_v(int v) const;
  bool h_is_locally_delaunay(int h) const;
  bool h_is_flippable(int h) const;
  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  utils::SimpleGenerator<int> generate_V_of_f(int f) const;
  utils::SimpleGenerator<int>
  generate_H_out_v_clockwise(int v, int h_start = -1) const;
  utils::SimpleGenerator<int> generate_H_right_f(int f) const;
  utils::SimpleGenerator<int> generate_H_rotcw_h(int h) const;
  utils::SimpleGenerator<int> generate_H_next_h(int h) const;
  utils::SimpleGenerator<int> generate_H_right_b(int b) const;
  utils::SimpleGenerator<int> generate_F_incident_v(int v) const;

  ///////////////////////////////////////////
  // Miscellaneous properties ///////////////
  ///////////////////////////////////////////
  /**
   * @brief Get the number of vertices in the mesh
   *
   * @return int
   */
  int num_vertices() const;
  /**
   * @brief Get the number edges in the mesh
   *
   * @return int
   */
  int num_edges() const;
  /**
   * @brief Get the numberhalf edges in the mesh
   *
   * @return int
   */
  int num_half_edges() const;
  /**
   * @brief Get the number of faces in the mesh
   *
   * @return int
   */
  int num_faces() const;
  /**
   * @brief Get the Euler characteristic of the mesh
   *
   * @return int
   */
  int euler_characteristic() const;
  /**
   * @brief Get the number of dicconnected boundary components
   *
   * @return int
   */
  int num_boundaries() const;
  /**
   * @brief Get the genus of the mesh
   *
   * @return int
   */
  int genus() const;

};

class

} // namespace mathutils
