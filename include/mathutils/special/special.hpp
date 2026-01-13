#pragma once

/**
 * @file special.hpp
 * @brief Special functions
 */

#include "mathutils/shared_utils.hpp"
#include "mathutils/simple_generator.hpp"
#include "mathutils/special/log_factorial_lookup_table.hpp"
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

/**
 * @brief Compute the index of (l,m) in [(0,0), (1,-1), (1,0), (1,1), (2,-2),
 * ...]
 *
 * @param l The degree (non-negative).
 * @param m The order (must be in the range [-l, l]).
 * @return int The spherical harmonic index.
 */
inline int spherical_harmonic_index_n_LM(int l, int m) {
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  return l * (l + 1) + m;
}

/**
 * @brief Compute the n-th element of the sequence [(0,0), (1,-1), (1,0), (1,1),
 * (2,-2),
 * ...]
 *
 * @param n The linear index (must be non-negative).
 * @return std::array<int, 2> The (l,m) pair.
 */
inline std::array<int, 2> spherical_harmonic_index_lm_N(int n) {
  if (n < 0) {
    throw std::out_of_range("n must be non-negative");
  }
  if (n > SPHERICAL_HARMONIC_INDEX_N_MAX) {
    int l = static_cast<int>(std::floor(std::sqrt(n)));
    return {l, n - l * (l + 1)};
  }
  return {SPHERICAL_HARMONIC_INDEX_LM_N_LOOKUP_TABLE[n][0],
          SPHERICAL_HARMONIC_INDEX_LM_N_LOOKUP_TABLE[n][1]};
}

/**
 * @brief Compute the associated Legendre polynomial with spherical harmonics
 * normalization \f$P_{m m}(\theta) = e^{-im\phi}Y_{m m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi/2\f$ and \f$m\geq 0\f$. Used to initialize the
 * recursion. Does not check the validity of \f$m\f$ or \f$\theta\f$.
 * @param m Degree/order of the polynomial (\f$m=0,1,\ldots\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi/2\f$).
 * @return double Value of the associated normalized Legendre polynomial.
 */
inline double reduced_spherical_Pmm(int m, double theta) {
  // if (theta == 0.0) {
  //   if (m == 0) {
  //     return 1.0 / std::sqrt(4 * M_PI);
  //   }
  //   return 0.0;
  // }
  // double sigma = (1 - 2 * (m & 1)); // this is just std::pow(-1, m);
  // return sigma * std::exp(m * std::log(std::sin(theta)) -
  //                         0.5 * std::log(4 * M_PI) - m * std::log(2) +
  //                         0.5 * std::lgamma(2 * m + 2) - std::lgamma(m + 1));
  return (theta == 0.0)
             ? (m == 0) / std::sqrt(4 * M_PI)
             : (1 - 2 * (m & 1)) *
                   std::exp(m * std::log(std::sin(theta)) -
                            0.5 * std::log(4 * M_PI) - m * std::log(2) +
                            0.5 * std::lgamma(2 * m + 2) - std::lgamma(m + 1));
}

/**
 * @brief Generate a sequence of associated Legendre polynomials with spherical
 * harmonics normalization \f$P_{\ell m}(\theta) = e^{-im\phi}Y_{\ell m}(\theta,
 * \phi)\f$ for fixed order \f$m=\ell_{start}\f$ and degree \f$\ell =
 * \ell_{start}, \ell_{start} + 1, \ldots, \ell_{end}\f$. Does not check the
 * validity of \f$\theta\f$.
 *
 * @param l_start Starting value \f$\ell\f$. Equivalent
 * @param l_end Final value of \f$\ell\f$.
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi/2\f$).
 */
mathutils::SimpleGenerator<double>
generate_reduced_spherical_Plm_recursion_three_term_upward_ell(int l_start,
                                                               int l_end,
                                                               double theta);

/**
 * @brief Generate a sequence of associated Legendre polynomials with spherical
 * harmonics normalization \f$P_{\ell m}(\theta) = e^{-im\phi}Y_{\ell m}(\theta,
 * \phi)\f$ for fixed order \f$m=\ell_{start}\f$ and degree \f$\ell =
 * \ell_{start}, \ell_{start} + 1, \ldots, \ell_{end}\f$. Yields pairs of (l,
 * P_ell_m). Does not check the validity of \f$\theta\f$.
 *
 * @param l_start Starting value \f$\ell\f$. Equivalent
 * @param l_end Final value of \f$\ell\f$.
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi/2\f$).
 */
mathutils::SimpleGenerator<std::pair<int, double>>
enumerate_reduced_spherical_Plm_recursion_three_term_upward_ell(int l_start,
                                                                int l_end,
                                                                double theta);

/**
 * @brief Compute the associated Legendre polynomial with spherical harmonics
 * normalization \f$P_{\ell m}(\theta) = e^{-im\phi}Y_{\ell m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi/2\f$ and \f$m\geq 0\f$. Does not check the validity
 * of inputs.
 *
 * @param l Degree of the polynomial (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the polynomial (\f$m=0,1,\ldots,\ell\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi/2\f$).
 * @return double Value of the associated normalized Legendre polynomial.
 */
inline double reduced_spherical_Plm(int l, int m, double theta) {
  // double cos_theta = std::cos(theta);
  // // P_{m m}
  // double P_ell_minus_one_m = reduced_spherical_Pmm(m, theta);
  // if (l == m) {
  //   return P_ell_minus_one_m;
  // }
  // // P_{m+1 m}
  // double P_ell_m = cos_theta * std::sqrt(2 * m + 3) * P_ell_minus_one_m;
  // for (int ell = m + 2; ell <= l; ++ell) {
  //   // double q =
  //   //     cos_theta *
  //   //         std::sqrt((2 * ell - 1) * (2 * ell + 1) / ((ell - m) * (ell +
  //   //         m))) * P_ell_m -
  //   //     std::sqrt((ell - m - 1) * (ell + m - 1) * (2 * ell + 1) /
  //   //               ((ell - m) * (ell + m) * (2 * ell - 3))) *
  //   //         P_ell_minus_one_m;
  //   const double den_ell_m = (ell - m) * 1.0 * (ell + m);
  //   const double c_ell_m =
  //       std::sqrt(((2.0 * ell - 1.0) * (2.0 * ell + 1.0)) / den_ell_m);
  //   const double den_ell_minus_one_m =
  //       (ell - m) * 1.0 * (ell + m) * (2.0 * ell - 3.0);
  //   const double c_ell_minus_one_m =
  //       std::sqrt(((ell - m - 1.0) * (ell + m - 1.0) * (2.0 * ell + 1.0)) /
  //                 den_ell_minus_one_m);
  //   const double q =
  //       cos_theta * c_ell_m * P_ell_m - c_ell_minus_one_m *
  //       P_ell_minus_one_m;
  //   P_ell_minus_one_m = P_ell_m;
  //   P_ell_m = q;
  // }
  // return P_ell_m;
  double P_ell_m{0.0};
  for (auto P : generate_reduced_spherical_Plm_recursion_three_term_upward_ell(
           m, l, theta)) {
    // ell_start = m
    // ell_end = l
    P_ell_m = P;
  }
  return P_ell_m;
}

/**
 * @brief Compute the associated Legendre polynomial with spherical harmonics
 * normalization \f$P_{\ell m}(\theta) = e^{-im\phi}Y_{\ell m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the polynomial (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the polynomial (\f$m=-\ell,\ldots,\ell\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi\f$).
 * @return double Value of the associated normalized Legendre polynomial.
 */
inline double spherical_Plm(int l, int m, double theta) {
  int abs_m = std::abs(m);
  // int sigma = 1;
  // if (m < 0) {
  //   sigma *= std::pow(-1, m);
  // }
  // if (theta > M_PI_2) {
  //   sigma *= std::pow(-1, l-abs_m);
  // }
  // sign difference between spherical_Plm and reduced_spherical_Plm
  // int sigma =
  //     ((m < 0) && (m & 1)) ^ ((theta > M_PI_2) && ((l - abs_m) & 1)) ? -1 :
  //     1;
  // // transform theta to [0, pi/2]
  // theta = std::min(theta, M_PI - theta);
  // m = abs_m;
  // return sigma * reduced_spherical_Plm(l, abs_m, theta);
  return ((m < 0) && (m & 1)) ^ ((theta > M_PI_2) && ((l - abs_m) & 1))
             ? -reduced_spherical_Plm(l, abs_m, std::min(theta, M_PI - theta))
             : reduced_spherical_Plm(l, abs_m, std::min(theta, M_PI - theta));
}

/**
 * @brief Compute the real spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the spherical harmonic (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the spherical harmonic (\f$m=-\ell,\ldots,\ell\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi\f$).
 * @param phi Azimuthal angle in radians (\f$0\leq\phi<2\pi\f$).
 * @return double Value of the real spherical harmonic.
 */
inline double recursive_real_Ylm(int l, int m, double theta, double phi) {
  // int abs_m = std::abs(m);
  // double Plm = spherical_Plm(l, abs_m, theta);
  // if (m < 0) {
  //   return (1 - 2 * (m & 1)) * std::sqrt(2) * std::sin(abs_m * phi) * Plm;
  // } else if (m > 0) {
  //   return (1 - 2 * (m & 1)) * std::sqrt(2) * std::cos(abs_m * phi) * Plm;
  // }
  // // m == 0
  // return Plm;
  if (m < 0) {
    return -std::sqrt(2) * std::sin(m * phi) * spherical_Plm(l, m, theta);
  } else if (m > 0) {
    return (1 - 2 * (m & 1)) * std::sqrt(2) * std::cos(m * phi) *
           spherical_Plm(l, m, theta);
  }
  // m == 0
  return spherical_Plm(l, m, theta);
}

/**
 * @brief Compute the real spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the spherical harmonic (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the spherical harmonic (\f$m=-\ell,\ldots,\ell\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi\f$).
 * @param phi Azimuthal angle in radians (\f$0\leq\phi<2\pi\f$).
 * @return double Value of the real spherical harmonic.
 */
inline double real_Ylm(int l, int m, double theta, double phi) {
  // check l and m
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta < 0 || theta > M_PI) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  return recursive_real_Ylm(l, m, theta, phi);
}

/**
 * @brief Compute the real spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the spherical harmonic (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the spherical harmonic (\f$m=-\ell,\ldots,\ell\f$).
 * @param thetaphi_coord_P Matrix containing the polar and azimuthal angles.
 * @return Eigen::VectorXd Values of the real spherical harmonics.
 */
Eigen::VectorXd real_Ylm(int l, int m, const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute real spherical harmonics \f$y_{l m}(\theta, \phi)\f$ for
 * \f$\ell=0,...,\ell_{max}\f$ at the given coordinates.
 *
 * @param l_max The maximum order of the spherical harmonics (non-negative
 * integer).
 * @param thetaphi_coord_P A matrix of shape (num_points, 2) where each row
 * contains the polar angle \f$\theta\f$ and azimuthal angle \f$\phi\f$ in
 * radians.
 * @return A matrix of shape (num_points, num_modes=l_max * (l_max + 2) + 1).
 * Mode index is related to (l,m) by
 * spherical_harmonic_index_n_LM/spherical_harmonic_index_lm_N functions.
 */
Eigen::MatrixXd compute_all_real_Ylm(int l_max,
                                     const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute the spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the spherical harmonic (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the spherical harmonic (\f$m=-\ell,\ldots,\ell\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi\f$).
 * @param phi Azimuthal angle in radians (\f$0\leq\phi<2\pi\f$).
 * @return std::complex<double> Value of the spherical harmonic.
 */
inline std::complex<double> recursive_Ylm(int l, int m, double theta,
                                          double phi) {
  return spherical_Plm(l, m, theta) *
         std::exp(std::complex<double>(0, m * phi));
}

/**
 * @brief Compute the spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the spherical harmonic (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the spherical harmonic (\f$m=-\ell,\ldots,\ell\f$).
 * @param theta Polar angle in radians (\f$0\leq\theta\leq\pi\f$).
 * @param phi Azimuthal angle in radians (\f$0\leq\phi<2\pi\f$).
 * @return std::complex<double> Value of the spherical harmonic.
 */
inline std::complex<double> Ylm(int l, int m, double theta, double phi) {
  // check l and m
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta < 0 || theta > M_PI) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  return recursive_Ylm(l, m, theta, phi);
}

/**
 * @brief Compute the spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l Degree of the spherical harmonic (\f$\ell=0,1,\ldots\f$).
 * @param m Order of the spherical harmonic (\f$m=-\ell,\ldots,\ell\f$).
 * @param thetaphi_coord_P Matrix containing the polar and azimuthal angles.
 * @return Eigen::VectorXcd Values of the spherical harmonics.
 */
Eigen::VectorXcd Ylm(int l, int m, const Eigen::MatrixXd &thetaphi_coord_P);

/**
 * @brief Compute the spherical harmonics \f$Y_{l m}(\theta, \phi)\f$
 * for \f$0\leq\theta\leq\pi\f$ and \f$-\ell \leq m\geq \ell\f$.
 *
 * @param l_max The maximum order of the spherical harmonics (non-negative
 * integer).
 * @param thetaphi_coord_P A matrix of shape (num_points, 2) where each row
 * contains the polar angle \f$\theta\f$ and azimuthal angle \f$\phi\f$ in
 * radians.
 * @return A matrix of shape (num_points, num_modes=l_max * (l_max + 2) + 1).
 * Mode index is related to (l,m) by
 * spherical_harmonic_index_n_LM/spherical_harmonic_index_lm_N functions.
 */
Eigen::MatrixXcd compute_all_Ylm(int l_max,
                                 const Eigen::MatrixXd &thetaphi_coord_P);

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
Eigen::MatrixXd fit_real_sh_coefficients_to_points(const Eigen::MatrixXd &XYZ0,
                                                   int l_max,
                                                   double reg_lambda = 0.0);

} // namespace special
} // namespace mathutils
