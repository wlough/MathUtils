#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <span>

namespace py = pybind11;

template <class T>
static py::array_t<T> numpy_view_span(std::span<T> s, py::object base) {
  return py::array_t<T>(
      py::buffer_info(s.data(), static_cast<py::ssize_t>(sizeof(T)),
                      py::format_descriptor<T>::format(), 1,
                      {static_cast<py::ssize_t>(s.size())},
                      {static_cast<py::ssize_t>(sizeof(T))}),
      std::move(base) // keep owner alive
  );
}
