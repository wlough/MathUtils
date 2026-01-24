// matrix_type_caster.hpp
#pragma once

#include "mathutils/matrix.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11::detail {

template <typename T> struct type_caster<mathutils::Matrix<T>> {
public:
  PYBIND11_TYPE_CASTER(mathutils::Matrix<T>, _("numpy.ndarray"));

  // Python -> C++ (copies)
  bool load(handle src, bool) {
    py::array a = py::array::ensure(src);
    if (!a || a.ndim() != 2)
      return false;

    py::array_t<T, py::array::c_style | py::array::forcecast> at(a);
    if (!at)
      return false;

    const std::size_t r = static_cast<std::size_t>(at.shape(0));
    const std::size_t c = static_cast<std::size_t>(at.shape(1));

    value = mathutils::Matrix<T>(r, c);
    if (value.size() != 0)
      std::memcpy(value.data(), at.data(), sizeof(T) * value.size());
    return true;
  }

  // // C++ -> Python (zero-copy by moving into heap-owned capsule)
  // static handle cast(mathutils::Matrix<T> src, return_value_policy, handle) {
  //   auto *heap = new mathutils::Matrix<T>(std::move(src));
  //   py::capsule base(heap, [](void *p) { delete
  //   reinterpret_cast<mathutils::Matrix<T> *>(p); });

  //   const ssize_t r = static_cast<ssize_t>(heap->rows());
  //   const ssize_t c = static_cast<ssize_t>(heap->cols());
  //   const ssize_t s0 = static_cast<ssize_t>(sizeof(T) * heap->cols());
  //   const ssize_t s1 = static_cast<ssize_t>(sizeof(T));

  //   py::array out(py::dtype::of<T>(), {r, c}, {s0, s1}, heap->data(), base);
  //   return out.release();
  // }
  // C++ -> Python (zero-copy by moving into heap-owned capsule)
  static handle cast(mathutils::Matrix<T> src, return_value_policy, handle) {
    auto *heap = new mathutils::Matrix<T>(std::move(src));
    py::capsule base(heap, [](void *p) {
      delete reinterpret_cast<mathutils::Matrix<T> *>(p);
    });

    ssize_t r = static_cast<ssize_t>(heap->rows());
    ssize_t c = static_cast<ssize_t>(heap->cols());

    // Default: (r, c) row-major view
    ssize_t out_r = r;
    ssize_t out_c = c;

    // If N×1, present as 1×N (row vector)
    if (c == 1 && r > 1) {
      out_r = 1;
      out_c = r;
    }

    const ssize_t s0 = static_cast<ssize_t>(sizeof(T) * out_c); // row stride
    const ssize_t s1 = static_cast<ssize_t>(sizeof(T));         // col stride

    py::array out(py::dtype::of<T>(), {out_r, out_c}, {s0, s1}, heap->data(),
                  base);
    return out.release();
  }
};

} // namespace pybind11::detail

// // Example usage:
// #include "matrix_type_caster.hpp"
// mathutils::Matrix<double> foo();
// // Python sees numpy.ndarray
// PYBIND11_MODULE(mathutils_backend, m) { m.def("foo", &foo); }