#pragma once

/**
 * @file funs.hpp
 * @brief Numerical functions
 */

#include "mathutils/log_factorial_lookup_table.hpp"
#include "mathutils/spherical_harmonics_index_lookup_table.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
// #include <complex>

/////////////////////////////////////
/////////////////////////////////////
// Simple non-vectorized functions //
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
/**
 * @brief Compute the sign (1,-1, or 0) of a value.
 *
 * @tparam T The type of the input value (must be comparable to T{0}).
 * @param x The input value.
 * @return The sign of the input value.
 */
template <typename T> inline int sign(T x) { return (x > T{0}) - (x < T{0}); }

} // namespace mathutils
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////

/////////////////////////////////////
/////////////////////////////////////
// Special functions ////////////////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace special {

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
 * \f[
 * \mathbb{R}\rightarrow
 * \mathbb{R}\cup\lbrace-\infty\rbrace\times\mathbb{Z}_{2}\\
 * \f]
 *
 * @param x The input value (can be floating-point or integral).
 * @return std::pair<double, int> {re_log, arg_over_pi}
 */
template <typename T>
inline std::pair<double, int> ReLogRe_ImLogRe_over_pi(const T &x) {
  if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
    return (x == T{0}) ? {-std::numeric_limits<double>::infinity(), 0}
                       : {std::log(std::abs(x)), (x < T{0})};
  } else {
    double Re_x = static_cast<double>(x);
    return (Re_x == 0.0) ? {-std::numeric_limits<double>::infinity(), 0}
                         : {std::log(std::abs(Re_x)), (Re_x < 0.0)};
  }
}

/////////////////////////
// Spherical harmonics //
/////////////////////////

inline int spherical_harmonic_index_n_LM(int l, int m) {
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  return l * (l + 1) + m;
}

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

inline int minus_one_to_pow(int n) {
  // Returns -1 if n is odd, 1 if n is even
  // return (n & 1) ? -1 : 1;
  return 1 - 2 * (n & 1);
}

inline double log_spherical_harmonic_normalization_Nklm(int k, int l, int m) {
  return -0.5 * std::log(4 * M_PI) - (std::abs(m) + 2 * k) * std::log(2) +
         0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + std::abs(m)) +
         0.5 * log_factorial(l - std::abs(m)) -
         log_factorial(l - std::abs(m) - 2 * k) -
         log_factorial(std::abs(m) + k) - log_factorial(k);
}

Eigen::MatrixXd
new_compute_all_real_Ylm(int l_max, const Eigen::MatrixXd &thetaphi_coord_P) {
  // check l_max
  if (l_max < 0) {
    throw std::out_of_range("l_max must be non-negative");
  }
  // check thetaphi_coord_P
  if (thetaphi_coord_P.cols() != 2) {
    throw std::invalid_argument("thetaphi_coord_P must have 2 columns");
  }
  int num_points = thetaphi_coord_P.rows();
  if (num_points <= 0) {
    throw std::invalid_argument("thetaphi_coord_P must have at least one row");
  }
  //////////////////////////////////
  int num_modes = l_max * (l_max + 2) + 1; // = n_LM(l_max, l_max)+1
  Eigen::MatrixXd Y(num_points, num_modes);
  Y.setZero();

  for (int p{0}; p < num_points; ++p) {
    double theta = thetaphi_coord_P(p, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(p, 1);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    // get ReLogRe_ImLogRe_over_pi for cos(theta) and sin(theta)
    auto [log_abs_cos_theta, arg_over_pi_cos_theta] =
        ReLogRe_ImLogRe_over_pi(cos_theta);
    auto [log_abs_sin_theta, arg_over_pi_sin_theta] =
        ReLogRe_ImLogRe_over_pi(sin_theta);
    for (int l{0}; l <= l_max; l++) {
      // m = 0 case
      int n = l * (l + 1);
      for (int k{0}; k <= l / 2; k++) {
        int ImLog_over_pi_Xklm = k + l * arg_over_pi_cos_theta;
        double ReLog_Xklm = -0.5 * std::log(4 * M_PI) - (2 * k) * std::log(2) +
                            0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l) +
                            0.5 * log_factorial(l) - log_factorial(l - 2 * k) -
                            log_factorial(k) - log_factorial(k) +
                            (l - 2 * k) * log_abs_cos_theta +
                            (2 * k) * log_abs_sin_theta;
        Y(p, n) += minus_one_to_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
      }
      for (int m{1}; m <= l; m++) {
        int n_plus = l * (l + 1) + m;
        int n_minus = l * (l + 1) - m;
        for (int k{0}; k <= (l - m) / 2; k++) {
          int ImLog_over_pi_Xklm =
              k + (l - m) * arg_over_pi_cos_theta + m * arg_over_pi_sin_theta;
          double ReLog_Xklm =
              -0.5 * std::log(4 * M_PI) - (m + 2 * k) * std::log(2) +
              0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + m) +
              0.5 * log_factorial(l - m) - log_factorial(l - m - 2 * k) -
              log_factorial(m + k) - log_factorial(k) +
              (l - m - 2 * k) * log_abs_cos_theta +
              (m + 2 * k) * log_abs_sin_theta;
          Y(p, n_plus) +=
              minus_one_to_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
        }
        Y(p, n_minus) = std::sqrt(2) * std::sin(m * phi) * Y(p, n_plus);
        Y(p, n_plus) *= std::sqrt(2) * std::cos(m * phi);
      }
    }
  }
  return Y;
}

/**
 * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for m>=0 and 0<=theta<=pi.
 * @param l Degree of the spherical harmonic.
 * @param m Order of the spherical harmonic.
 * @param theta Polar angle in radians.
 * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
 */
inline double phi_independent_Ylm(int l, int m, double theta) {
  double epsilon = 1e-8;
  double minus_log4 = std::log(0.25);

  int abs_m = std::abs(m);

  int ell_minus_m = l - abs_m;
  double cos_theta = std::cos(theta);
  double abs_cos_theta = std::abs(cos_theta);
  // double sin_theta = std::sin(theta); // always positive for theta in [0, pi]
  double abs_sin_theta = std::sin(theta); // std::abs(sin_theta);

  double rlm = 1.0 / (2.0 * std::sqrt(M_PI));
  if ((m > 0) && (abs_m % 2 == 1)) {
    rlm *= -1;
  }
  // if ((m < 0) && (abs_m % 2 == 1)) {
  //   rlm *= -1;
  // }
  // if (abs_m % 2 == 1) {
  //   rlm *= -1;
  // }
  if ((cos_theta < 0) && (ell_minus_m % 2 == 1)) {
    rlm *= -1;
  }
  int k_bound = (ell_minus_m) / 2;
  int sk = 1;
  // double log_qk;
  // double sum_Qk;
  double log_qk = 0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) -
                  abs_m * std::log(2) - 0.5 * log_factorial(ell_minus_m) -
                  log_factorial(abs_m);

  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  // If cos(theta) ~ +/-sin(theta) replace sin(theta) with tan(theta)
  if (std::abs(abs_cos_theta - abs_sin_theta) < epsilon) {
    double tan_theta = std::tan(theta);
    double abs_tan_theta = std::abs(tan_theta);
    double log_abs_cos_theta = std::log(abs_cos_theta);
    double log_abs_tan_theta = std::log(abs_tan_theta);
    log_qk += l * log_abs_cos_theta + abs_m * log_abs_tan_theta;
    double sum_Qk = sk * std::exp(log_qk);
    for (int k = 1; k <= k_bound; ++k) {
      sk *= -1;
      log_qk += 2 * log_abs_tan_theta;
      log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
                std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
                std::log(k);
      sum_Qk += sk * std::exp(log_qk);
    }
    return rlm * sum_Qk;
  }

  // if cos(theta) ~ 0, sin(theta) ~ 1
  if (abs_cos_theta < epsilon) {
    double log_abs_sin_theta = std::log(abs_sin_theta);
    log_qk += abs_m * log_abs_sin_theta;
    double abs_cos_theta_pow = std::pow(abs_cos_theta, ell_minus_m);
    double sum_Qk = sk * std::exp(log_qk) * abs_cos_theta_pow;
    for (int k = 1; k <= k_bound; ++k) {
      sk *= -1;
      log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
                std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
                std::log(k);
      abs_cos_theta_pow = std::pow(abs_cos_theta, ell_minus_m - 2 * k);
      sum_Qk += sk * std::exp(log_qk) * abs_cos_theta_pow;
    }

    return rlm * sum_Qk;
  }

  // if sin(theta) ~ 0, cos(theta) ~ +/-1
  if (abs_sin_theta < epsilon) {
    // double log_abs_cos_theta = std::log(abs_cos_theta);
    // log_qk += ell_minus_m * log_abs_cos_theta;
    // double abs_sin_theta_pow = std::pow(abs_sin_theta, abs_m);
    // double sum_Qk = sk * std::exp(log_qk);
    // for (int k = 1; k <= k_bound; ++k) {
    //   sk *= -1;
    //   log_qk += -2 * log_abs_cos_theta;
    //   log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
    //             std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
    //             std::log(k);
    //   abs_sin_theta_pow = std::pow(abs_sin_theta, abs_m + 2 * k);
    //   sum_Qk += sk * std::exp(log_qk) * abs_sin_theta_pow;
    // }
    // return rlm * sum_Qk;
    // double log_abs_cos_theta = std::log(abs_cos_theta);
    // log_qk += ell_minus_m * log_abs_cos_theta;
    double trig_term =
        std::pow(abs_cos_theta, ell_minus_m) * std::pow(abs_sin_theta, abs_m);
    double sum_Qk = sk * std::exp(log_qk) * trig_term;
    for (int k = 1; k <= k_bound; ++k) {
      sk *= -1;
      log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
                std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
                std::log(k);
      trig_term = std::pow(abs_cos_theta, ell_minus_m - 2 * k) *
                  std::pow(abs_sin_theta, abs_m + 2 * k);
      sum_Qk += sk * std::exp(log_qk) * trig_term;
    }
    return rlm * sum_Qk;
  }

  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////
  // if log(cos(theta)) and log(sin(theta)) are defined
  double log_abs_cos_theta = std::log(abs_cos_theta);
  double log_abs_sin_theta = std::log(abs_sin_theta);
  log_qk += ell_minus_m * log_abs_cos_theta + abs_m * log_abs_sin_theta;
  double sum_Qk = sk * std::exp(log_qk);
  for (int k = 1; k <= k_bound; ++k) {
    sk *= -1;
    log_qk += -2 * log_abs_cos_theta + 2 * log_abs_sin_theta;
    log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
              std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
              std::log(k);
    sum_Qk += sk * std::exp(log_qk);
  }

  return rlm * sum_Qk;
}

inline double real_Ylm(int l, int m, double theta, double phi) {

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

Eigen::MatrixXd compute_all_real_Ylm(int l_max,
                                     const Eigen::MatrixXd &thetaphi_coord_P) {
  // check l_max
  if (l_max < 0) {
    throw std::out_of_range("l_max must be non-negative");
  }
  // check thetaphi_coord_P
  if (thetaphi_coord_P.cols() != 2) {
    throw std::invalid_argument("thetaphi_coord_P must have 2 columns");
  }
  int num_points = thetaphi_coord_P.rows();
  if (num_points <= 0) {
    throw std::invalid_argument("thetaphi_coord_P must have at least one row");
  }
  //////////////////////////////////
  int num_modes = l_max * (l_max + 2) + 1; // = n_LM(l_max, l_max)+1
  Eigen::MatrixXd Y(num_points, num_modes);
  Y.setZero();
  for (int l{0}; l <= l_max; l++) {
    for (int m{-l}; m <= l; m++) {
      int n = l * (l + 1) + m;
      for (int p{0}; p < num_points; ++p) {
        double theta = thetaphi_coord_P(p, 0);
        double phi = thetaphi_coord_P(p, 1);
        Y(p, n) = real_Ylm(l, m, theta, phi);
      }
    }
  }
  return Y;
  //////////////////////////////////
  //////////////////////////////////

  //////////////////////////////////
  //////////////////////////////////
  //////////////////////////////////
  //////////////////////////////////
  // double epsilon = 1e-8;
  // double minus_log4 = std::log(0.25);

  // Eigen::VectorXd theta_P = thetaphi_coord_P.col(0);
  // Eigen::VectorXd phi_P = thetaphi_coord_P.col(1);
  // Eigen::VectorXd cos_theta_P = theta_P.array().cos();
  // Eigen::VectorXd abs_cos_theta_P = cos_theta_P.cwiseAbs();
  // Eigen::VectorXd sin_theta_P = theta_P.array().sin();
  // Eigen::VectorXd abs_sin_theta_P = sin_theta_P.cwiseAbs();

  // int num_modes = l_max * (l_max + 2) + 1; // = n_LM(l_max, l_max)+1

  // // // Y = Eigen::MatrixXd::Zero(num_points, num_modes);
  // Eigen::MatrixXd Y(num_points, num_modes);
  // Y.setZero();

  // for (int l{0}; l <= l_max; ++l) {
  //   for (int m{-l}; m <= l; ++m) {
  //     int n = l * (l + 1) + m;
  //     int sign_m = (m > 0) - (m < 0);
  //     int abs_m = std::abs(m);
  //     int ell_minus_m = l - abs_m;
  //     int ell_plus_m = l + abs_m;
  //     int k_bound = (ell_minus_m) / 2;
  //     double log_qk0 = 0.5 * std::log(2 * l + 1) +
  //                      0.5 * log_factorial(l + abs_m) - abs_m * std::log(2) -
  //                      0.5 * log_factorial(l - abs_m) - log_factorial(abs_m);
  //     for (int p{0}; p < num_points; ++p) {

  //       int s_lmtheta = 1;
  //       int sign_cos_theta = (theta_P[p] < M_PI_2) - (theta_P[p] > M_PI_2);
  //       if (l % 2 == 1) {
  //         s_lmtheta *= sign_cos_theta;
  //       }
  //       // if (m % 2 == 1) {
  //       //   s_lmtheta *= -sign_m *sign_cos_theta;
  //       // }
  //       // if (m % 2 == 1) {
  //       //   s_lmtheta *= -sign_cos_theta;
  //       // }
  //       // if (m % 2 == 1) {
  //       //   s_lmtheta *= -1;
  //       // }
  //       if (m % 2 == 1) {
  //         s_lmtheta *= sign_cos_theta;
  //       }

  //       int s_k = 1;
  //       double log_qk = log_qk0;
  //       double sum_Qk{0.0};
  //       /////////////////////////////////////////
  //       /////////////////////////////////////////
  //       /////////////////////////////////////////
  //       // If cos(theta) ~ +/-sin(theta) replace sin(theta) with tan(theta)
  //       if (std::abs(abs_cos_theta_P[p] - abs_sin_theta_P[p]) < epsilon) {
  //         double tan_theta = std::tan(theta_P[p]);
  //         double abs_tan_theta = std::abs(tan_theta);
  //         double log_abs_cos_theta = std::log(abs_cos_theta_P[p]);
  //         double log_abs_tan_theta = std::log(abs_tan_theta);
  //         log_qk += l * log_abs_cos_theta + abs_m * log_abs_tan_theta;
  //         sum_Qk = s_k * std::exp(log_qk);
  //         for (int k{1}; k <= k_bound; k++) {
  //           s_k *= -1;
  //           log_qk += 2 * log_abs_tan_theta;
  //           log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
  //                     std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k)
  //                     - std::log(k);
  //           sum_Qk += s_k * std::exp(log_qk);
  //         }
  //       } else if (abs_cos_theta_P[p] <
  //                  epsilon) { // if cos(theta) ~ 0, sin(theta) ~ 1
  //         double log_abs_sin_theta = std::log(abs_sin_theta_P[p]);
  //         log_qk += abs_m * log_abs_sin_theta;
  //         double abs_cos_theta_pow = std::pow(abs_cos_theta_P[p],
  //         ell_minus_m); double sum_Qk = s_k * std::exp(log_qk) *
  //         abs_cos_theta_pow; for (int k = 1; k <= k_bound; ++k) {
  //           s_k *= -1;
  //           log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
  //                     std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k)
  //                     - std::log(k);
  //           abs_cos_theta_pow =
  //               std::pow(abs_cos_theta_P[p], ell_minus_m - 2 * k);
  //           sum_Qk += s_k * std::exp(log_qk) * abs_cos_theta_pow;
  //         }
  //       } else if (abs_sin_theta_P[p] <
  //                  epsilon) { // if sin(theta) ~ 0, cos(theta) ~ +/-1
  //         double trig_term = std::pow(abs_cos_theta_P[p], ell_minus_m) *
  //                            std::pow(abs_sin_theta_P[p], abs_m);
  //         double sum_Qk = s_k * std::exp(log_qk) * trig_term;
  //         for (int k = 1; k <= k_bound; ++k) {
  //           s_k *= -1;
  //           log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
  //                     std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k)
  //                     - std::log(k);
  //           trig_term = std::pow(abs_cos_theta_P[p], ell_minus_m - 2 * k) *
  //                       std::pow(abs_sin_theta_P[p], abs_m + 2 * k);
  //           sum_Qk += s_k * std::exp(log_qk) * trig_term;
  //         }
  //       } else { // if log(cos(theta)) and log(sin(theta)) are defined
  //         double log_abs_cos_theta = std::log(abs_cos_theta_P[p]);
  //         double log_abs_sin_theta = std::log(abs_sin_theta_P[p]);
  //         log_qk += ell_minus_m * log_abs_cos_theta + abs_m *
  //         log_abs_sin_theta; sum_Qk = s_k * std::exp(log_qk); for (int k{1};
  //         k <= k_bound; ++k) {
  //           s_k *= -1;
  //           log_qk += -2 * log_abs_cos_theta + 2 * log_abs_sin_theta;
  //           log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
  //                     std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k)
  //                     - std::log(k);
  //           sum_Qk += s_k * std::exp(log_qk);
  //         }
  //       }
  //       /////////////////////////////////////////
  //       /////////////////////////////////////////
  //       /////////////////////////////////////////
  //       Y(p, n) = s_lmtheta * sum_Qk / (2.0 * std::sqrt(M_PI));
  //       if (m < 0) {
  //         Y(p, n) *= std::sqrt(2) * std::sin(abs_m * phi_P[p]);
  //       } else if (m > 0) {
  //         Y(p, n) *= std::sqrt(2) * std::cos(abs_m * phi_P[p]);
  //       }
  //     }
  //   }
  // }
  //
  // return Y;
}
} // namespace special
} // namespace mathutils
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
