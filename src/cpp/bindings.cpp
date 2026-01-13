// bindings.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

void bind_shared_utils(py::module_ &m);

void bind_findiff(py::module_ &m);
void bind_mesh(py::module_ &m);
void bind_random(py::module_ &m);
void bind_special(py::module_ &m);

PYBIND11_MODULE(mathutils_backend, m) {
  m.doc() = "PyMathUtils backend";

  bind_shared_utils(m);

  auto m_findiff = m.def_submodule("findiff");
  bind_findiff(m_findiff);

  auto m_mesh = m.def_submodule("mesh");
  bind_mesh(m_mesh);

  auto m_random = m.def_submodule("random");
  bind_random(m_random);

  auto m_special = m.def_submodule("special");
  bind_special(m_special);
}
