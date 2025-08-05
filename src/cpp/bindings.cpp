// #include "mathutils/funs.hpp"
// #include "mathutils/log_factorial_lookup_table.hpp"
// #include "mathutils/spherical_harmonics_index_lookup_table.hpp"
// #include <pybind11/complex.h>
// #include <pybind11/eigen.h>
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// namespace py = pybind11;

// PYBIND11_MODULE(mathutils_backend, m) {
//   m.doc() = "Fast mathematical utilities backend";

//   m.def("log_factorial", &mathutils::special::log_factorial,
//         "Compute log factorial of n", py::arg("n"));

//   m.def("spherical_harmonic_index_n_LM",
//         &mathutils::special::spherical_harmonic_index_n_LM, "...",
//         py::arg("l"), py::arg("m"));

//   m.def("spherical_harmonic_index_lm_N",
//         &mathutils::special::spherical_harmonic_index_lm_N, "...",
//         py::arg("n"));

//   m.def("Ylm",
//         static_cast<std::complex<double> (*)(int, int, double, double)>(
//             &mathutils::special::Ylm),
//         "Compute spherical harmonic Ylm for single point", py::arg("l"),
//         py::arg("m"), py::arg("theta"), py::arg("phi"));

//   m.def("Ylm",
//         static_cast<Eigen::VectorXcd (*)(int, int, const Eigen::MatrixXd &)>(
//             &mathutils::special::Ylm),
//         "Compute spherical harmonic Ylm for multiple points", py::arg("l"),
//         py::arg("m"), py::arg("thetaphi_coord_P"));

//   m.def("real_Ylm",
//         static_cast<double (*)(int, int, double, double)>(
//             &mathutils::special::real_Ylm),
//         "Compute real spherical harmonic Ylm for single point", py::arg("l"),
//         py::arg("m"), py::arg("theta"), py::arg("phi"));

//   m.def("real_Ylm",
//         static_cast<Eigen::VectorXd (*)(int, int, const Eigen::MatrixXd &)>(
//             &mathutils::special::real_Ylm),
//         "Compute real spherical harmonic Ylm for multiple points",
//         py::arg("l"), py::arg("m"), py::arg("thetaphi_coord_P"));

//   m.def("compute_all_real_Ylm", &mathutils::special::compute_all_real_Ylm,
//         "...", py::arg("l_max"), py::arg("thetaphi_coord_P"));

//   m.def("old_compute_all_real_Ylm",
//         &mathutils::special::old_compute_all_real_Ylm, "...",
//         py::arg("l_max"), py::arg("thetaphi_coord_P"));

//   m.def("compute_all_Ylm", &mathutils::special::compute_all_Ylm, "...",
//         py::arg("l_max"), py::arg("thetaphi_coord_P"));

//   // Version info
//   m.attr("__version__") = "0.1.5";
// }

#include "mathutils/funs.hpp"
#include "mathutils/log_factorial_lookup_table.hpp"
#include "mathutils/spherical_harmonics_index_lookup_table.hpp"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(mathutils_backend, m) {
  m.doc() = "Fast mathematical utilities backend";

  // Basic utility functions (in main module)
  m.def("sign", &mathutils::sign<double>, "sign function (double)",
        py::arg("x"));
  m.def("sign", &mathutils::sign<int>, "sign function (int)", py::arg("x"));

  // Create a submodule for special functions
  py::module_ special =
      m.def_submodule("special", "Special mathematical functions");

  special.def("log_factorial", &mathutils::special::log_factorial,
              "Compute log factorial of n", py::arg("n"));

  special.def("spherical_harmonic_index_n_LM",
              &mathutils::special::spherical_harmonic_index_n_LM,
              "Convert (l,m) to linear index", py::arg("l"), py::arg("m"));

  special.def("spherical_harmonic_index_lm_N",
              &mathutils::special::spherical_harmonic_index_lm_N,
              "Convert linear index to (l,m)", py::arg("n"));

  // Spherical harmonics in special submodule
  special.def("Ylm",
              static_cast<std::complex<double> (*)(int, int, double, double)>(
                  &mathutils::special::Ylm),
              "Compute spherical harmonic Ylm for single point", py::arg("l"),
              py::arg("m"), py::arg("theta"), py::arg("phi"));

  special.def(
      "Ylm",
      static_cast<Eigen::VectorXcd (*)(int, int, const Eigen::MatrixXd &)>(
          &mathutils::special::Ylm),
      "Compute spherical harmonic Ylm for multiple points", py::arg("l"),
      py::arg("m"), py::arg("thetaphi_coord_P"));

  special.def("real_Ylm",
              static_cast<double (*)(int, int, double, double)>(
                  &mathutils::special::real_Ylm),
              "Compute real spherical harmonic Ylm for single point",
              py::arg("l"), py::arg("m"), py::arg("theta"), py::arg("phi"));

  special.def(
      "real_Ylm",
      static_cast<Eigen::VectorXd (*)(int, int, const Eigen::MatrixXd &)>(
          &mathutils::special::real_Ylm),
      "Compute real spherical harmonic Ylm for multiple points", py::arg("l"),
      py::arg("m"), py::arg("thetaphi_coord_P"));

  // Batch computation functions in special submodule
  special.def("compute_all_real_Ylm", &mathutils::special::compute_all_real_Ylm,
              "Compute all real Ylm up to l_max", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));

  special.def("old_compute_all_real_Ylm",
              &mathutils::special::old_compute_all_real_Ylm,
              "Old implementation of compute_all_real_Ylm", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));

  special.def("compute_all_Ylm", &mathutils::special::compute_all_Ylm,
              "Compute all complex Ylm up to l_max", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));

  // Constants/lookup tables
  //   m.attr("LOG_FACTORIAL_LOOKUP_TABLE_SIZE") =
  //   LOG_FACTORIAL_LOOKUP_TABLE_SIZE; m.attr("SPHERICAL_HARMONIC_INDEX_N_MAX")
  //   = SPHERICAL_HARMONIC_INDEX_N_MAX;

  // Version info
  m.attr("__version__") = "0.1.8";
}