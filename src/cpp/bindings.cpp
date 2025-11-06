#include "mathutils/findiff.hpp"
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
  m.def("thetaphi_from_xyz", &mathutils::thetaphi_from_xyz,
        "Convert Cartesian coordinates (x, y, z) to unit sphere coordinates "
        "(theta, phi)",
        py::arg("xyz"));
  m.def("xyz_from_rthetaphi", &mathutils::xyz_from_rthetaphi,
        "Convert spherical coordinates (r, theta, phi) to Cartesian "
        "coordinates (x, y, z)",
        py::arg("rthetaphi"));
  m.def("rthetaphi_from_xyz", &mathutils::rthetaphi_from_xyz,
        "Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, "
        "theta, phi)",
        py::arg("xyz"));

  // Create a submodule for special functions
  py::module_ special =
      m.def_submodule("special", "Special mathematical functions");
  //   special.def("reduced_spherical_Pmm",
  //               &mathutils::special::reduced_spherical_Pmm,
  //               "Compute reduced spherical Pmm", py::arg("m"),
  //               py::arg("theta"));
  //   special.def("reduced_spherical_Plm",
  //               &mathutils::special::reduced_spherical_Plm,
  //               "Compute reduced spherical Plm", py::arg("l"), py::arg("m"),
  //               py::arg("theta"));
  //   special.def("spherical_Plm", &mathutils::special::spherical_Plm,
  //               "Compute spherical Plm", py::arg("l"), py::arg("m"),
  //               py::arg("theta"));
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
  special.def("spherical_harmonic_index_n_LM",
              &mathutils::special::spherical_harmonic_index_n_LM,
              "Convert (l,m) to linear index", py::arg("l"), py::arg("m"));
  special.def("spherical_harmonic_index_lm_N",
              &mathutils::special::spherical_harmonic_index_lm_N,
              "Convert linear index to (l,m)", py::arg("n"));
  special.def("compute_all_real_Ylm", &mathutils::special::compute_all_real_Ylm,
              "Compute all real Ylm up to l_max", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));
  special.def("compute_all_Ylm", &mathutils::special::compute_all_Ylm,
              "Compute all complex Ylm up to l_max", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));
  special.def("fit_real_sh_coefficients_to_points",
              &mathutils::special::fit_real_sh_coefficients_to_points,
              "Fit real spherical harmonic coefficients to points",
              py::arg("XYZ0"), py::arg("l_max"), py::arg("reg_lambda"));

  //////////////////////////////////
  //////////////////////////////////
  //////////////////////////////////
  special.def("minus_one_to_int_pow", &mathutils::special::minus_one_to_int_pow,
              "Compute (-1)^n for integer n", py::arg("n"));

  special.def("ReLogRe_ImLogRe_over_pi",
              &mathutils::special::ReLogRe_ImLogRe_over_pi<double>,
              "Compute ReLogRe and ImLogRe over pi for a given value (double)",
              py::arg("x"));
  special.def("ReLogRe_ImLogRe_over_pi",
              &mathutils::special::ReLogRe_ImLogRe_over_pi<int>,
              "Compute ReLogRe and ImLogRe over pi for a given value (int)",
              py::arg("x"));

  special.def("log_factorial", &mathutils::special::log_factorial,
              "Compute log factorial of n", py::arg("n"));

  special.def("series_Ylm",
              static_cast<std::complex<double> (*)(int, int, double, double)>(
                  &mathutils::special::series_Ylm),
              "Compute spherical harmonic Ylm for single point", py::arg("l"),
              py::arg("m"), py::arg("theta"), py::arg("phi"));
  special.def(
      "series_Ylm",
      static_cast<Eigen::VectorXcd (*)(int, int, const Eigen::MatrixXd &)>(
          &mathutils::special::series_Ylm),
      "Compute spherical harmonic Ylm for multiple points", py::arg("l"),
      py::arg("m"), py::arg("thetaphi_coord_P"));

  special.def("series_real_Ylm",
              static_cast<double (*)(int, int, double, double)>(
                  &mathutils::special::series_real_Ylm),
              "Compute real spherical harmonic Ylm for single point",
              py::arg("l"), py::arg("m"), py::arg("theta"), py::arg("phi"));
  special.def(
      "series_real_Ylm",
      static_cast<Eigen::VectorXd (*)(int, int, const Eigen::MatrixXd &)>(
          &mathutils::special::series_real_Ylm),
      "Compute real spherical harmonic Ylm for multiple points", py::arg("l"),
      py::arg("m"), py::arg("thetaphi_coord_P"));

  special.def("compute_all_series_real_Ylm",
              &mathutils::special::compute_all_series_real_Ylm,
              "Compute all real Ylm up to l_max", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));

  special.def("old_compute_all_real_Ylm",
              &mathutils::special::old_compute_all_real_Ylm,
              "Old implementation of compute_all_real_Ylm", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));

  special.def("compute_all_series_Ylm",
              &mathutils::special::compute_all_series_Ylm,
              "Compute all complex Ylm up to l_max", py::arg("l_max"),
              py::arg("thetaphi_coord_P"));

  // Create a submodule for finite differences
  py::module_ findiff =
      m.def_submodule("findiff", "Finite difference utilities");

  findiff.def(
      "fornberg_weights",
      [](py::EigenDRef<const Eigen::VectorXd> x_stencil, double x0, int d_max) {
        // Call the Eigen overload directly
        return mathutils::findiff::fornberg_weights(
            x_stencil, x0,
            d_max); // returns Eigen::MatrixXd
      },
      py::arg("x_stencil"), py::arg("x0"), py::arg("d_max"),
      R"doc(
Compute Fornberg finite-difference weights up to order d_max at x0.

Parameters
----------
x_stencil : array_like (n,)
    Stencil node coordinates (need not be uniform).
x0 : float
    Evaluation point.
d_max : int
    Maximum derivative order (>= 0).

Returns
-------
W : ndarray, shape (d_max+1, n)
    Weight matrix; W[d, j] multiplies f(x_stencil[j]) to approximate the d-th derivative at x0.

Notes
-----
This is a thin wrapper over the C++ Eigen implementation; it throws on invalid input.
)doc");

  // Helper converters
  auto np1d_to_vecd = [](py::array x) {
    if (x.ndim() != 1)
      throw std::invalid_argument("x must be 1D");
    py::array_t<double, py::array::c_style | py::array::forcecast> a(x);
    std::vector<double> v(a.size());
    auto r = a.unchecked<1>();
    for (ssize_t i = 0; i < a.size(); ++i)
      v[static_cast<size_t>(i)] = r(i);
    return v;
  };
  auto np1d_to_veci = [](py::array s) {
    if (s.ndim() != 1)
      throw std::invalid_argument("stencil must be 1D");
    py::array_t<long long, py::array::c_style | py::array::forcecast> a(s);
    std::vector<int> v(a.size());
    auto r = a.unchecked<1>();
    for (ssize_t i = 0; i < a.size(); ++i)
      v[static_cast<size_t>(i)] = static_cast<int>(r(i));
    return v;
  };
  auto np1d_opt_to_veci =
      [&](py::object obj) -> std::optional<std::vector<int>> {
    if (obj.is_none())
      return std::nullopt;
    return np1d_to_veci(obj.cast<py::array>());
  };

  py::class_<mathutils::findiff::FiniteDifference1D>(findiff,
                                                     "FiniteDifference1D")
      .def(py::init<>())
      // Constructor-like builder
      .def_static(
          "build_from",
          [&](py::array x, py::array stencil, int deriv_order,
              py::object period, py::object b0, py::object b1) {
            mathutils::findiff::BuildSpec spec;
            spec.x = np1d_to_vecd(x);
            spec.stencil = np1d_to_veci(stencil);
            spec.deriv_order = deriv_order;
            if (!period.is_none())
              spec.period = period.cast<double>();
            spec.boundary0_stencil = np1d_opt_to_veci(b0);
            spec.boundary1_stencil = np1d_opt_to_veci(b1);
            mathutils::findiff::FiniteDifference1D op;
            {
              py::gil_scoped_release nogil;
              op.build(spec);
            }
            return op;
          },
          py::arg("x"), py::arg("stencil"), py::arg("deriv_order"),
          py::arg("period") = py::none(),
          py::arg("boundary0_stencil") = py::none(),
          py::arg("boundary1_stencil") = py::none(),
          R"doc(Build a finite-difference operator from Fornberg weights.)doc")
      .def_static(
          "build_periodic",
          [&](py::array x, py::array stencil, int deriv_order,
              py::object period) {
            mathutils::findiff::BuildSpecPeriodic spec;
            spec.x = np1d_to_vecd(x);
            spec.stencil = np1d_to_veci(stencil);
            spec.deriv_order = deriv_order;
            spec.period = period.cast<double>();
            mathutils::findiff::FiniteDifference1D op;
            {
              py::gil_scoped_release nogil;
              op.build(spec);
            }
            return op;
          },
          py::arg("x"), py::arg("stencil"), py::arg("deriv_order"),
          py::arg("period"),
          R"doc(Build a finite-difference operator from Fornberg weights on periodic domain.)doc")
      .def_static(
          "build_nonperiodic",
          [&](py::array x, py::array stencil, int deriv_order, py::object b0,
              py::object b1) {
            mathutils::findiff::BuildSpecNonPeriodic spec;
            spec.x = np1d_to_vecd(x);
            spec.interior_stencil = np1d_to_veci(stencil);
            spec.deriv_order = deriv_order;
            spec.boundary0_stencil = np1d_to_veci(b0);
            spec.boundary1_stencil = np1d_to_veci(b1);
            mathutils::findiff::FiniteDifference1D op;
            {
              py::gil_scoped_release nogil;
              op.build(spec);
            }
            return op;
          },
          py::arg("x"), py::arg("stencil"), py::arg("deriv_order"),
          py::arg("boundary0_stencil"), py::arg("boundary1_stencil"),
          R"doc(Build a finite-difference operator from Fornberg weights on nonperiodic domain.)doc")
      .def("size", &mathutils::findiff::FiniteDifference1D::size)
      .def(
          "apply",
          [](const mathutils::findiff::FiniteDifference1D &op, py::array y) {
            if (y.ndim() == 1) {
              if (y.shape(0) != op.size())
                throw std::invalid_argument("y length mismatch");
              py::array_t<double> out(y.shape(0));
              {
                py::gil_scoped_release nogil;
                op.apply(y.cast<py::array_t<double>>().data(),
                         out.mutable_data());
              }
              return out;
            }
            if (y.ndim() == 2 && y.shape(0) == op.size()) {
              const int Nx = static_cast<int>(y.shape(0));
              const int nvec = static_cast<int>(y.shape(1));
              py::array_t<double> out({Nx, nvec});
              {
                py::gil_scoped_release nogil;
                auto Y = y.cast<py::array_t<double>>();
                op.apply_batch(Y.data(), Nx, out.mutable_data(), Nx, nvec);
              }
              return out;
            }
            throw std::invalid_argument("y must be 1D (Nx,) or 2D (Nx, nvec)");
          },
          py::arg("y"),
          R"doc(Apply the operator to y. Supports shape (Nx,) or (Nx, nvec).)doc")
      .def(
          "csr_triplets",
          [](const mathutils::findiff::FiniteDifference1D &op) {
            std::vector<mathutils::findiff::Index64> I, J;
            std::vector<double> V;
            op.triplets(I, J, V);
            return py::make_tuple(
                py::array_t<mathutils::findiff::Index64>(I.size(), I.data()),
                py::array_t<mathutils::findiff::Index64>(J.size(), J.data()),
                py::array_t<double>(V.size(), V.data()),
                py::make_tuple(op.size(), op.size()));
          },
          R"doc(Return (I, J, V, shape) to build a SciPy csr_matrix if
        desired.)doc");
}