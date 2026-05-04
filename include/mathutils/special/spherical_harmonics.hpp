#pragma once

/**
 * @file spherical_harmonics.hpp
 * @brief Spherical harmonics
 */

// #include "mathutils/shared_utils.hpp"
#include "mathutils/matrix.hpp"
#include "mathutils/simple_generator.hpp"
#include "mathutils/special/spherical_harmonics_index_lookup_table.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace mathutils {
namespace special {

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
Matrix<double> real_Ylm(int l, int m, const Matrix<double> &thetaphi_coord_P);
// Eigen::VectorXd real_Ylm(int l, int m, const Eigen::MatrixXd
// &thetaphi_coord_P);

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
Matrix<std::complex<double>> Ylm(int l, int m,
                                 const Matrix<double> &thetaphi_coord_P);

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
Matrix<double> compute_all_real_Ylm(int l_max,
                                    const Matrix<double> &thetaphi_coord_P);

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
Matrix<std::complex<double>>
compute_all_Ylm(int l_max, const Matrix<double> &thetaphi_coord_P);
} // namespace special
} // namespace mathutils
