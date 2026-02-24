// #include "mathutils/mesh/half_edge_mesh.hpp"
// #include "mathutils/mesh/mesh_common.hpp"
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
//
// namespace py = pybind11;
//
// static py::array X_ambient_V_view(mathutils::mesh::HalfEdgeMesh &self) {
//   using mathutils::mesh::Real;
//   auto &M = self.X_ambient_V_;
//
//   // shape and strides for row-major contiguous (r,c)
//   std::array<ssize_t, 2> shape = {static_cast<ssize_t>(M.rows()),
//                                   static_cast<ssize_t>(M.cols())};
//   std::array<ssize_t, 2> strides = {
//       static_cast<ssize_t>(sizeof(Real) * M.cols()),
//       static_cast<ssize_t>(sizeof(Real))};
//
//   // base object keeps `self` alive while the ndarray exists
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                              // ptr
//                       static_cast<ssize_t>(sizeof(Real)),    // itemsize
//                       py::format_descriptor<Real>::format(), // dtype
//                       2,                                     // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array h_out_V_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.h_out_V_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("h_out_V_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array h_directed_E_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.h_directed_E_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("h_directed_E_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array h_right_F_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.h_right_F_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("h_right_F_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array h_negative_B_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.h_negative_B_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("h_negative_B_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array v_origin_H_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.v_origin_H_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("v_origin_H_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array e_undirected_H_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.e_undirected_H_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("e_undirected_H_view: expected an N-by-1
//     matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array f_left_H_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.f_left_H_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("f_left_H_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array h_next_H_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.h_next_H_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("h_next_H_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
//
// static py::array h_twin_H_view(mathutils::mesh::HalfEdgeMesh &self) {
//   auto &M = self.topo.h_twin_H_;
//
//   using Index = mathutils::mesh::Index;
//
//   if (M.cols() != 1) {
//     throw std::runtime_error("h_twin_H_view: expected an N-by-1 matrix");
//   }
//
//   // 1D NumPy view: shape (N,), contiguous stride
//   std::array<ssize_t, 1> shape = {static_cast<ssize_t>(M.rows())};
//   std::array<ssize_t, 1> strides = {static_cast<ssize_t>(sizeof(Index))};
//
//   // Keep self alive while NumPy holds the view
//   py::object base = py::cast(&self);
//
//   return py::array(
//       py::buffer_info(M.data(),                               // ptr
//                       static_cast<ssize_t>(sizeof(Index)),    // itemsize
//                       py::format_descriptor<Index>::format(), // dtype
//                       1,                                      // ndim
//                       shape, strides),
//       base);
// }
