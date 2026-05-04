#pragma once

/**
 * @file special.hpp
 * @brief Special functions
 */

#include "mathutils/matrix.hpp"
#include "mathutils/shared_utils.hpp"
#include "mathutils/simple_generator.hpp"
#include "mathutils/special/log_factorial_lookup_table.hpp"
#include "mathutils/special/spherical_harmonics.hpp"
#include "mathutils/special/spherical_harmonics_index_lookup_table.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <complex>
#include <coroutine>
#include <cstdint>
#include <stdexcept>

/////////////////////////////////////
/////////////////////////////////////
// Special functions ////////////////
/////////////////////////////////////
/////////////////////////////////////

// Some inline template functions
namespace mathutils {
namespace special {
/**
 * @brief Compute \f$(-1)^n\f$ for integer n.
 */
inline int minus_one_to_int_pow(int n) {
  // Returns -1 if n is odd, 1 if n is even
  return 1 - 2 * (n & 1);
}

/**
 * @brief Compute \f$\ln(n!)\f$ for a non-negative integer n.
 * Uses a lookup table for small values and lgamma for larger values.
 *
 * @param n The non-negative integer for which to compute log(n!).
 * @return The logarithm of the factorial of n.
 */
inline double log_factorial(uint64_t n) {
  return (n < LOG_FACTORIAL_LOOKUP_TABLE_SIZE)
             ? LOG_FACTORIAL_LOOKUP_TABLE[n]
             : std::lgamma(static_cast<double>(n + 1)); // lgamma(n+1) = log(n!)
}

/**
 * @brief The Heaviside step function. Returns 0.0 for negative values, 1.0 for
 * positive values, and 0.5 for zero.
 *
 * @tparam T The type of the input value (must be comparable to T{0}).
 * @param x The input value.
 * @return The Heaviside step function value.
 */
template <typename T> inline double Heaviside(T x) {
  return 0.5 + 0.5 * ((x > T{0}) - (x < T{0}));
}

/**
 * @brief Compute the real part of the complex logarithm \f$Re(Log(x))=
 * ln(|x|)\f$. Returns \f$Re(Log(0))=-\infty\f$ for \f$x=0\f$.
 *
 * \f[\mathbb{C}\rightarrow \mathbb{R}\cup\lbrace-\infty\rbrace\\
 * x\mapsto Re(Log(x))
 * \f]
 *
 * @param x The input value (can be floating-point or integral).
 * @return The logarithm of the input value.
 */
template <typename T> inline double ReLog(const T &x) {
  return (x == T{0}) ? -std::numeric_limits<double>::infinity()
                     : std::log(std::abs(x));
}

/**
 * @brief Compute \f$Re(Log(Re(x)))\f$ and \f$Im(Log(Re(x)))/\pi\f$
 *
 * @param x The input value (can be floating-point or integral).
 * @return std::pair<double, int> {re_log, arg_over_pi}
 */
template <typename T>
inline std::pair<double, int> ReLogRe_ImLogRe_over_pi(const T &x) {
  if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
    // return (x == T{0}) ? {-std::numeric_limits<double>::infinity(), 0}
    //                    : {std::log(std::abs(x)), static_cast<int>(x < T{0})};
    return (x == T{0})
               ? std::make_pair(-std::numeric_limits<double>::infinity(), 0)
               : std::make_pair(std::log(std::abs(x)),
                                static_cast<int>(x < T{0}));
  } else {
    double Re_x = static_cast<double>(x);
    // return (Re_x == 0.0)
    //            ? {-std::numeric_limits<double>::infinity(), 0}
    //            : {std::log(std::abs(Re_x)), static_cast<int>(Re_x < 0.0)};
    return (Re_x == 0.0)
               ? std::make_pair(-std::numeric_limits<double>::infinity(), 0)
               : std::make_pair(std::log(std::abs(Re_x)),
                                static_cast<int>(Re_x < 0.0));
  }
}

template <typename T>
inline std::pair<double, int> ReLogRe_ImLogRe_over_pi_of_x_to_pow(const T &x,
                                                                  int n) {
  if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
    return (n == 0) ? std::make_pair(0.0, 0)
           : (x == T{0})
               ? std::make_pair(-n * std::numeric_limits<double>::infinity(), 0)
               : std::make_pair(n * std::log(std::abs(x)),
                                static_cast<int>(x < T{0} && (n & 1)));
  } else {
    double Re_x = static_cast<double>(x);
    return (n == 0) ? std::make_pair(0.0, 0)
           : (Re_x == 0.0)
               ? std::make_pair(-n * std::numeric_limits<double>::infinity(), 0)
               : std::make_pair(n * std::log(std::abs(Re_x)),
                                static_cast<int>(Re_x < 0.0 && (n & 1)));
  }
}
} // namespace special
} // namespace mathutils

namespace mathutils {
namespace special {

/////////////////////////
// Spherical harmonics //
/////////////////////////

//////////////////////////////////////////////////////
// Alternative implementations of spherical harmonics.
// May be unstable for large l and m.

/**
 * @brief Compute real spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * using series expansion in cos(theta) and sin(theta). Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
double series_real_Ylm(int l, int m, double theta, double phi);

/**
 * @brief Compute real spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * using series expansion in cos(theta) and sin(theta). Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
Eigen::VectorXd series_real_Ylm(int l, int m,
                                const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * using series expansion in cos(theta) and sin(theta). Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
std::complex<double> series_Ylm(int l, int m, double theta, double phi);

/**
 * @brief Compute spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * using series expansion in cos(theta) and sin(theta). Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
Eigen::VectorXcd series_Ylm(int l, int m,
                            const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute real spherical harmonics \f$y_{l m}(\theta, \phi)\f$ for
 * \f$\ell=0,...,\ell_{max}\f$ at the given coordinates. Unstable for large
 * \f$l\f$ and \f$m\f$.
 *
 * @param l_max The maximum order of the spherical harmonics (non-negative
 * integer).
 * @param thetaphi_coord_P A matrix of shape (num_points, 2) where each row
 * contains the polar angle \f$\theta\f$ and azimuthal angle \f$\phi\f$ in
 * radians.
 * @return A matrix of shape (num_points, l_max * (l_max + 2) + 1)
 */
Eigen::MatrixXd
compute_all_series_real_Ylm(int l_max, const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute spherical harmonics \f$Y_{l m}(\theta, \phi)\f$ for
 * \f$\ell=0,...,\ell_{max}\f$ at the given coordinates. Unstable for large
 * \f$l\f$ and \f$m\f$.
 *
 * @param l_max The maximum order of the spherical harmonics (non-negative
 * integer).
 * @param thetaphi_coord_P A matrix of shape (num_points, 2) where each row
 * contains the polar angle \f$\theta\f$ and azimuthal angle \f$\phi\f$ in
 * radians.
 * @return A matrix of shape (num_points, l_max * (l_max + 2) + 1)
 */
Eigen::MatrixXcd
compute_all_series_Ylm(int l_max, const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for m>=0 and 0<=theta<=pi.
 * Unstable for large \f$l\f$
 *
 * @param l Degree of the spherical harmonic.
 * @param m Order of the spherical harmonic.
 * @param theta Polar angle in radians.
 * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
 */
double phi_independent_Ylm(int l, int m, double theta);

/**
 * @brief Compute spherical harmonics \f$Y_{l m}(\theta, \phi)\f$ for
 * \f$\ell=0,...,\ell_{max}\f$ at the given coordinates. Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
inline std::complex<double> old_Ylm(int l, int m, double theta, double phi) {
  if (theta > M_PI || theta < 0.0) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  return std::exp(std::complex<double>(0, m * phi)) *
         phi_independent_Ylm(l, m, theta);
}

/**
 * @brief Compute real spherical harmonics \f$y_{l m}(\theta, \phi)\f$ for
 * \f$\ell=0,...,\ell_{max}\f$ at the given coordinates. Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
inline double old_real_Ylm(int l, int m, double theta, double phi) {

  if (theta > M_PI || theta < 0.0) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  int abs_m = std::abs(m);
  double val = phi_independent_Ylm(l, abs_m, theta);
  if (abs_m % 2 == 1) {
    val *= -1;
  }
  if (m < 0) {
    return val * std::sqrt(2) * std::sin(abs_m * phi);
  }
  if (m == 0) {
    return val;
  }
  return val * std::sqrt(2) * std::cos(abs_m * phi);
}

/**
 * @brief Compute real spherical harmonics \f$y_{l m}(\theta, \phi)\f$ for
 * \f$\ell=0,...,\ell_{max}\f$ at the given coordinates. Unstable for large
 * \f$l\f$ and \f$m\f$.
 */
Eigen::MatrixXd
old_compute_all_real_Ylm(int l_max, const Eigen::MatrixXd &thetaphi_coord_P);

} // namespace special
} // namespace mathutils

////////////////////////////////
// Spherical harmonic fitting //
////////////////////////////////
namespace mathutils {
namespace special {

/**
 * @brief ***BROKEN*** Solve for real spherical-harmonic coefficients that best
 * fit a 3-D point cloud on a sphere, with optional Dirichlet smoothing.
 *
 * Solves the Tikhonov system
 *     (Yᵀ Y + λ Lᵀ L) A = Yᵀ X,
 * where
 *     * Y  – design matrix of real harmonics, shape (N, (l_max+1)^2)
 *     * L  – diagonal matrix with entries l(l+1) (Dirichlet energy weight)
 *     * X  – input Cartesian coordinates, shape (N,3)
 * The result A(i,:) gives the (x,y,z) coefficients for harmonic index *i*.
 *
 * @param XYZ0        (N,3) matrix of Cartesian samples.
 * @param l_max       Highest harmonic degree to include.
 * @param reg_lambda  Non-negative Tikhonov parameter (λ = 0 ⇒ plain LS).
 * @return            ((l_max+1)^2, 3) matrix of expansion coefficients.
 */
Matrix<double> fit_real_sh_coefficients_to_points(const Matrix<double> &XYZ,
                                                  int l_max,
                                                  double reg_lambda = 0.0);

} // namespace special
} // namespace mathutils
