#include "mathutils/matrix.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
static py::array matrix_view(mathutils::Matrix<T> &M, py::handle owner) {
  py::object base = py::reinterpret_borrow<py::object>(owner);
  // 2D row-major view
  const ssize_t rows = static_cast<ssize_t>(M.rows());
  const ssize_t cols = static_cast<ssize_t>(M.cols());
  return py::array(
      py::buffer_info(
          M.data(),                           // ptr
          static_cast<ssize_t>(sizeof(T)),    // itemsize
          py::format_descriptor<T>::format(), // dtype
          2,                                  // ndim
          std::vector<ssize_t>{rows, cols},   // shape
          std::vector<ssize_t>{
              static_cast<ssize_t>(sizeof(T) * M.cols()),
              static_cast<ssize_t>(sizeof(T))}), // row-major strides
      base);
}

template <typename T>
static py::array vector_view(mathutils::Matrix<T> &M, py::handle owner) {
  py::object base = py::reinterpret_borrow<py::object>(owner);
  // 1D view for Nx1 or 1xN when numpy_view_ == Ndarray1D
  const ssize_t n = static_cast<ssize_t>(M.size());
  return py::array(py::buffer_info(M.data(),                        // ptr
                                   static_cast<ssize_t>(sizeof(T)), // itemsize
                                   py::format_descriptor<T>::format(), // dtype
                                   1,                                  // ndim
                                   std::vector<ssize_t>{n},            // shape
                                   std::vector<ssize_t>{static_cast<ssize_t>(
                                       sizeof(T))}), // strides
                   base);
}
