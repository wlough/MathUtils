// matrix_type_caster.hpp
#pragma once

/**
 * @file Defines pybind11 caster for Matrix<T> variants and numpy ndarray
 */

#include "mathutils/matrix.hpp"
#include "mathutils/mesh/mesh_common.hpp"
#include "mathutils/mesh/mesh_plyio.hpp"
#include <cstring>
#include <iostream> // DEBUG
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11::detail {

// template <typename T> struct type_caster<mathutils::Matrix<T>> {
// public:
//   PYBIND11_TYPE_CASTER(mathutils::Matrix<T>, _("numpy.ndarray"));

//   // Python -> C++ (copies)
//   // bool load(handle src, bool) {
//   //   py::array a = py::array::ensure(src);
//   //   if (!a || a.ndim() != 2)
//   //     return false;

//   //   py::array_t<T, py::array::c_style | py::array::forcecast> at(a);
//   //   if (!at)
//   //     return false;

//   //   const std::size_t r = static_cast<std::size_t>(at.shape(0));
//   //   const std::size_t c = static_cast<std::size_t>(at.shape(1));

//   //   value = mathutils::Matrix<T>(r, c);
//   //   if (value.size() != 0)
//   //     std::memcpy(value.data(), at.data(), sizeof(T) * value.size());
//   //   return true;
//   // }
//   // Python -> C++ (copies)
//   bool load(handle src, bool) {
//     py::array a = py::array::ensure(src);
//     if (!a)
//       return false;

//     // Accept either (r,c) or (N,)
//     if (a.ndim() == 2) {
//       py::array_t<T, py::array::c_style | py::array::forcecast> at(a);
//       if (!at)
//         return false;

//       const std::size_t r = static_cast<std::size_t>(at.shape(0));
//       const std::size_t c = static_cast<std::size_t>(at.shape(1));

//       value = mathutils::Matrix<T>(r, c, mathutils::NumpyView::Ndarray2D);
//       if (value.size() != 0)
//         std::memcpy(value.data(), at.data(), sizeof(T) * value.size());

//       return true;
//     }

//     if (a.ndim() == 1) {
//       py::array_t<T, py::array::c_style | py::array::forcecast> at(a);
//       if (!at)
//         return false;

//       const std::size_t n = static_cast<std::size_t>(at.shape(0));

//       //  (N,) -> Matrix(N,1)
//       value = mathutils::Matrix<T>(n, 1, mathutils::NumpyView::Ndarray1D);
//       if (value.size() != 0)
//         std::memcpy(value.data(), at.data(), sizeof(T) * value.size());

//       // record that this came from a 1D view.
//       // value.set_numpy_view(mathutils::NumpyView::Ndarray1D);
//       return true;
//     }

//     return false;
//   }

//   // C++ -> Python (zero-copy by moving into heap-owned capsule)
//   static handle cast(mathutils::Matrix<T> src, return_value_policy, handle) {

//     auto *heap = new mathutils::Matrix<T>(std::move(src));
//     py::capsule base(heap, [](void *p) {
//       delete reinterpret_cast<mathutils::Matrix<T> *>(p);
//     });
//     bool want_1d = heap->want_numpy_vector();

//     // // DEBUG
//     // bool want_1d_src = src.want_numpy_vector();
//     // if (want_1d != want_1d_src) {
//     //   throw std::runtime_error("want_1d mismatch between src and heap");
//     // }
//     // // DEBUG

//     if (want_1d) {
//       const ssize_t n = static_cast<ssize_t>(heap->size()); // N for N×1 or
//       1×N const ssize_t s0 = static_cast<ssize_t>(sizeof(T));

//       py::array out(py::dtype::of<T>(), {n}, {s0}, heap->data(), base);
//       return out.release();
//     } else {
//       const ssize_t r = static_cast<ssize_t>(heap->rows());
//       const ssize_t c = static_cast<ssize_t>(heap->cols());
//       const ssize_t s0 =
//           static_cast<ssize_t>(sizeof(T) * heap->cols()); // row-major
//       const ssize_t s1 = static_cast<ssize_t>(sizeof(T));

//       py::array out(py::dtype::of<T>(), {r, c}, {s0, s1}, heap->data(),
//       base); return out.release();
//     }
//   }
// };

template <typename T> struct type_caster<mathutils::Matrix<T>> {
public:
  PYBIND11_TYPE_CASTER(mathutils::Matrix<T>, _("numpy.ndarray"));

  // Python -> C++ (copies)
  // Note: pybind11 assumes output data is stored in `value`
  bool load(handle src, bool) {
    // Only accept objects that can be viewed as numpy array
    py::array arr_in = py::array::ensure(src);
    if (!arr_in)
      return false;
    // Only accept 1D or 2D
    auto ndim = arr_in.ndim();
    if (ndim != 1 && ndim != 2)
      return false;
    // Require exact dtype so std::variant can try other T's.
    // Stuff gets cast to wrong data types without this.
    if (!py::dtype::of<T>().is(arr_in.dtype()))
      return false;

    if (ndim == 2) {
      // Require C-contiguous (row-major) layout. No forcecast.
      py::array_t<T, py::array::c_style> at(arr_in);
      // py::array_t<T, py::array::c_style | py::array::forcecast> at(arr_in);
      if (!at)
        return false;

      const std::size_t num_rows = static_cast<std::size_t>(at.shape(0));
      const std::size_t num_cols = static_cast<std::size_t>(at.shape(1));
      value = mathutils::Matrix<T>(num_rows, num_cols,
                                   mathutils::NumpyView::Ndarray2D);
      if (value.size() != 0)
        std::memcpy(value.data(), at.data(), sizeof(T) * value.size());
      return true;
    }
    // else if (ndim == 1)
    // Require C-contiguous (row-major) layout. No forcecast.
    py::array_t<T, py::array::c_style> at(arr_in);
    if (!at)
      return false;

    const std::size_t num_vals = static_cast<std::size_t>(at.shape(0));

    value = mathutils::Matrix<T>(num_vals, 1, mathutils::NumpyView::Ndarray1D);

    // Ensure Matrix remembers "this was 1D"
    // value.set_numpy_view(mathutils::NumpyView::Ndarray1D);
    if (value.size() != 0)
      std::memcpy(value.data(), at.data(), sizeof(T) * value.size());
    return true;
  }

  // C++ -> Python (zero-copy by moving into heap-owned capsule)
  // return false upon failure
  static handle cast(mathutils::Matrix<T> src, return_value_policy, handle) {

    auto *heap = new mathutils::Matrix<T>(std::move(src));
    py::capsule base(heap, [](void *p) {
      delete reinterpret_cast<mathutils::Matrix<T> *>(p);
    });
    bool want_1d = heap->want_numpy_vector();

    if (want_1d) {
      const ssize_t n = static_cast<ssize_t>(heap->size()); // N for N×1 or 1×N
      const ssize_t s0 = static_cast<ssize_t>(sizeof(T));

      py::array out(py::dtype::of<T>(), {n}, {s0}, heap->data(), base);
      return out.release();
    } else {
      const ssize_t r = static_cast<ssize_t>(heap->rows());
      const ssize_t c = static_cast<ssize_t>(heap->cols());
      const ssize_t s0 =
          static_cast<ssize_t>(sizeof(T) * heap->cols()); // row-major
      const ssize_t s1 = static_cast<ssize_t>(sizeof(T));

      py::array out(py::dtype::of<T>(), {r, c}, {s0, s1}, heap->data(), base);
      return out.release();
    }
  }
};

} // namespace pybind11::detail

// // Example usage:
// #include "matrix_type_caster.hpp"
// mathutils::Matrix<double> foo();
// // Python sees numpy.ndarray
// PYBIND11_MODULE(mathutils_backend, m) { m.def("foo", &foo); }