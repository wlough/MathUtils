// bind_special.cpp
#include "mathutils/bind/bind_special.hpp"
#include "mathutils/special/special.hpp"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_special(py::module_ &m) {

  m.doc() = "Special functions.";

  ///////////////////////////////////////////
  // Special functions submodule `special` //
  ///////////////////////////////////////////

  //   m.def("reduced_spherical_Pmm",
  //               &mathutils::special::reduced_spherical_Pmm,
  //               "Compute reduced spherical Pmm", py::arg("m"),
  //               py::arg("theta"));
  //   m.def("reduced_spherical_Plm",
  //               &mathutils::special::reduced_spherical_Plm,
  //               "Compute reduced spherical Plm", py::arg("l"), py::arg("m"),
  //               py::arg("theta"));
  //   m.def("spherical_Plm", &mathutils::special::spherical_Plm,
  //               "Compute spherical Plm", py::arg("l"), py::arg("m"),
  //               py::arg("theta"));
  m.def("Ylm",
        static_cast<std::complex<double> (*)(int, int, double, double)>(
            &mathutils::special::Ylm),
        "Compute spherical harmonic Ylm for single point", py::arg("l"),
        py::arg("m"), py::arg("theta"), py::arg("phi"));
  m.def("Ylm",
        static_cast<Eigen::VectorXcd (*)(int, int, const Eigen::MatrixXd &)>(
            &mathutils::special::Ylm),
        "Compute spherical harmonic Ylm for multiple points", py::arg("l"),
        py::arg("m"), py::arg("thetaphi_coord_P"));
  m.def("real_Ylm",
        static_cast<double (*)(int, int, double, double)>(
            &mathutils::special::real_Ylm),
        "Compute real spherical harmonic Ylm for single point", py::arg("l"),
        py::arg("m"), py::arg("theta"), py::arg("phi"));
  m.def("real_Ylm",
        static_cast<Eigen::VectorXd (*)(int, int, const Eigen::MatrixXd &)>(
            &mathutils::special::real_Ylm),
        "Compute real spherical harmonic Ylm for multiple points", py::arg("l"),
        py::arg("m"), py::arg("thetaphi_coord_P"));
  m.def("spherical_harmonic_index_n_LM",
        &mathutils::special::spherical_harmonic_index_n_LM,
        "Convert (l,m) to linear index", py::arg("l"), py::arg("m"));
  m.def("spherical_harmonic_index_lm_N",
        &mathutils::special::spherical_harmonic_index_lm_N,
        "Convert linear index to (l,m)", py::arg("n"));
  m.def("compute_all_real_Ylm", &mathutils::special::compute_all_real_Ylm,
        "Compute all real Ylm up to l_max", py::arg("l_max"),
        py::arg("thetaphi_coord_P"));
  m.def("compute_all_Ylm", &mathutils::special::compute_all_Ylm,
        "Compute all complex Ylm up to l_max", py::arg("l_max"),
        py::arg("thetaphi_coord_P"));
  m.def("fit_real_sh_coefficients_to_points",
        &mathutils::special::fit_real_sh_coefficients_to_points,
        "Fit real spherical harmonic coefficients to points", py::arg("XYZ0"),
        py::arg("l_max"), py::arg("reg_lambda"));

  //////////////////////////////////
  //////////////////////////////////
  //////////////////////////////////
  m.def("minus_one_to_int_pow", &mathutils::special::minus_one_to_int_pow,
        "Compute (-1)^n for integer n", py::arg("n"));

  m.def("ReLogRe_ImLogRe_over_pi",
        &mathutils::special::ReLogRe_ImLogRe_over_pi<double>,
        "Compute ReLogRe and ImLogRe over pi for a given value (double)",
        py::arg("x"));
  m.def("ReLogRe_ImLogRe_over_pi",
        &mathutils::special::ReLogRe_ImLogRe_over_pi<int>,
        "Compute ReLogRe and ImLogRe over pi for a given value (int)",
        py::arg("x"));

  m.def("log_factorial", &mathutils::special::log_factorial,
        "Compute log factorial of n", py::arg("n"));

  m.def("series_Ylm",
        static_cast<std::complex<double> (*)(int, int, double, double)>(
            &mathutils::special::series_Ylm),
        "Compute spherical harmonic Ylm for single point", py::arg("l"),
        py::arg("m"), py::arg("theta"), py::arg("phi"));
  m.def("series_Ylm",
        static_cast<Eigen::VectorXcd (*)(int, int, const Eigen::MatrixXd &)>(
            &mathutils::special::series_Ylm),
        "Compute spherical harmonic Ylm for multiple points", py::arg("l"),
        py::arg("m"), py::arg("thetaphi_coord_P"));

  m.def("series_real_Ylm",
        static_cast<double (*)(int, int, double, double)>(
            &mathutils::special::series_real_Ylm),
        "Compute real spherical harmonic Ylm for single point", py::arg("l"),
        py::arg("m"), py::arg("theta"), py::arg("phi"));
  m.def("series_real_Ylm",
        static_cast<Eigen::VectorXd (*)(int, int, const Eigen::MatrixXd &)>(
            &mathutils::special::series_real_Ylm),
        "Compute real spherical harmonic Ylm for multiple points", py::arg("l"),
        py::arg("m"), py::arg("thetaphi_coord_P"));

  m.def("compute_all_series_real_Ylm",
        &mathutils::special::compute_all_series_real_Ylm,
        "Compute all real Ylm up to l_max", py::arg("l_max"),
        py::arg("thetaphi_coord_P"));

  m.def("old_compute_all_real_Ylm",
        &mathutils::special::old_compute_all_real_Ylm,
        "Old implementation of compute_all_real_Ylm", py::arg("l_max"),
        py::arg("thetaphi_coord_P"));

  m.def("compute_all_series_Ylm", &mathutils::special::compute_all_series_Ylm,
        "Compute all complex Ylm up to l_max", py::arg("l_max"),
        py::arg("thetaphi_coord_P"));
}
