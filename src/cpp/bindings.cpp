#include "mathutils/funs.hpp"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(mathutils_backend, m) {
  m.doc() = "Fast mathematical utilities backend";

  // Expose the log_factorial function
  m.def("log_factorial", &mathutils::log_factorial,
        "Compute log factorial of n", py::arg("n"));

  m.def("spherical_harmonic_index_n_LM",
        &mathutils::spherical_harmonic_index_n_LM, "...", py::arg("l"),
        py::arg("m"));

  m.def("spherical_harmonic_index_lm_N",
        &mathutils::spherical_harmonic_index_lm_N, "...", py::arg("n"));

  m.def("Ylm_vectorized", &mathutils::Ylm_vectorized, "...", py::arg("l"),
        py::arg("m"), py::arg("theta"), py::arg("phi"));

  m.def("real_Ylm_vectorized", &mathutils::real_Ylm_vectorized, "...",
        py::arg("l"), py::arg("m"), py::arg("theta"), py::arg("phi"));

  m.def("magnitude_Ylm", &mathutils::magnitude_Ylm, "...", py::arg("l"),
        py::arg("abs_m"), py::arg("abs_cos_theta"));

  // Expose the LOG factorial lookup table as read-only
  py::array_t<double> log_factorial_table =
      py::cast(mathutils::LOG_FACTORIAL_LOOKUP_TABLE);
  m.attr("LOG_FACTORIAL_LOOKUP_TABLE") = log_factorial_table;

  // // If you also want to expose regular factorial table:
  // py::array_t<uint64_t> factorial_table =
  //     py::cast(mathutils::FACTORIAL_LOOKUP_TABLE);
  // m.attr("FACTORIAL_LOOKUP_TABLE") = factorial_table;

  // Version info
  m.attr("__version__") = "0.1.0";
}