#pragma once

/**
 * @file funs.hpp
 * @brief Numerical functions and constants.
 */

#include "mathutils/lookup_tables.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
// #include <complex>

namespace mathutils {

template <typename T> inline int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

inline double log_factorial(uint64_t n) {
  if (n < 50) {
    return LOG_FACTORIAL_LOOKUP_TABLE[n];

  } else {
    // lgamma(n+1) = log(n!)
    return std::lgamma(n + 1);
  }
}

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
  int l = static_cast<int>(std::floor(std::sqrt(n)));
  return {l, n - l * (l + 1)};
}

/**
 * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for small sin(theta) and
 * cos(theta).
 * @param l Degree of the spherical harmonic.
 * @param m Order of the spherical harmonic.
 * @param theta Polar angle in radians.
 * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
 */
double phi_independent_problem_angle_small_Ylm(int l, int m, double theta) {
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);
  int abs_m = std::abs(m);
  double rlm = 1.0 / (2.0 * std::sqrt(M_PI));
  if (m > 0) {
    if (m % 2 == 1) {
      rlm *= -1;
    }
  }

  double log_qk = 0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) -
                  abs_m * std::log(2) - 0.5 * log_factorial(l - abs_m) -
                  log_factorial(abs_m);
  double trig_term =
      std::pow(cos_theta, l - abs_m) * std::pow(sin_theta, abs_m);
  double sk = 1;
  double sum_Qk = sk * std::exp(log_qk) * trig_term;
  double minus_log4 = std::log(0.25);

  for (int k = 1; k <= (l - abs_m) / 2; ++k) {
    sk *= -1;
    log_qk += minus_log4 + std::log(l - abs_m - 2 * k + 2) +
              std::log(l - abs_m - 2 * k + 1) - std::log(abs_m + k) -
              std::log(k);
    trig_term = std::pow(cos_theta, l - abs_m - 2 * k) *
                std::pow(sin_theta, abs_m + 2 * k);

    sum_Qk += sk * std::exp(log_qk) * trig_term;
  }

  return rlm * sum_Qk;
}

/**
 * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for sin(theta) ~
 * +-cos(theta).
 * @param l Degree of the spherical harmonic.
 * @param m Order of the spherical harmonic.
 * @param theta Polar angle in radians.
 * @param phi Azimuthal angle in radians.
 * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
 */
double phi_independent_problem_angle_same_Ylm(int l, int m, double theta,
                                              double phi) {
  double cos_theta = std::cos(theta);
  double tan_theta = std::tan(theta);
  double abs_cos_theta = std::abs(cos_theta);
  double abs_tan_theta = std::abs(tan_theta);
  double log_abs_cos_theta = std::log(abs_cos_theta);
  double log_abs_tan_theta = std::log(abs_tan_theta);
  int abs_m = std::abs(m);
  double trig_pows = std::pow(cos_theta, l) * std::pow(tan_theta, abs_m);
  double rlm = sign(trig_pows) / (2 * std::sqrt(M_PI));
  if (m > 0) {
    if (m % 2 == 1) {
      rlm *= -1;
    }
  }

  double log_qk = 0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) +
                  l * log_abs_cos_theta + abs_m * log_abs_tan_theta -
                  abs_m * std::log(2) - 0.5 * log_factorial(l - abs_m) -
                  log_factorial(abs_m);
  double sk = 1;
  double sum_Qk = sk * std::exp(log_qk);
  double minus_log4 = std::log(0.25);
  for (int k = 1; k <= (l - abs_m) / 2; ++k) {
    sk *= -1;
    log_qk +=
        2 * log_abs_tan_theta + minus_log4 + std::log(l - abs_m - 2 * k + 2) +
        std::log(l - abs_m - 2 * k + 1) - std::log(abs_m + k) - std::log(k);
    sum_Qk += sk * std::exp(log_qk);
  }
  return rlm * sum_Qk;
}

double _phi_independent_Ylm(int l, int m, double theta, double phi) {
  double cos_theta = std::cos(theta);
  double abs_cos_theta = std::abs(cos_theta);
  double epsilon = 1e-8;
  if (abs_cos_theta < epsilon) {
    return 0.0;
  }
  double sin_theta = std::sin(theta);
  double abs_sin_theta = std::abs(sin_theta);
  if (abs_sin_theta < epsilon) {
    return 0.0;
  }
  if (std::abs(abs_cos_theta - abs_sin_theta) < epsilon) {
    return 0.0;
  }
  double log_abs_cos_theta = std::log(abs_cos_theta);
  double log_abs_sin_theta = std::log(abs_sin_theta);
  int abs_m = std::abs(m);

  double trig_pows =
      std::pow(cos_theta, l - abs_m) * std::pow(sin_theta, abs_m);

  double rlm = sign(trig_pows) / (2 * std::sqrt(M_PI));
  // if abs_m % 2 == 1:
  if (m > 0) {
    if (m % 2 == 1) {
      rlm *= -1;
    }
  }

  double log_qk = (0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) +
                   (l - abs_m) * log_abs_cos_theta + abs_m * log_abs_sin_theta -
                   abs_m * std::log(2) - 0.5 * log_factorial(l - abs_m) -
                   log_factorial(abs_m));
  double sk = 1;
  double sum_Qk = sk * std::exp(log_qk);
  double minus_log4 = std::log(0.25);

  for (int k = 1; k <= (l - abs_m) / 2; ++k) {
    sk *= -1;
    log_qk +=
        (-2 * log_abs_cos_theta + 2 * log_abs_sin_theta + minus_log4 +
         std::log(l - abs_m - 2 * k + 2) + std::log(l - abs_m - 2 * k + 1) -
         std::log(abs_m + k) - std::log(k));
    sum_Qk += sk * std::exp(log_qk);
  }

  return rlm * sum_Qk;
}

/**
 * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for m>=0 and 0<=theta<=pi.
 * @param l Degree of the spherical harmonic.
 * @param m Order of the spherical harmonic.
 * @param theta Polar angle in radians.
 * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
 */
double phi_independent_Ylm(int l, int m, double theta) {
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

std::complex<double> Ylm(int l, int m, double theta, double phi) {

  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta > M_PI || theta < 0.0) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  return std::exp(std::complex<double>(0, m * phi)) *
         phi_independent_Ylm(l, m, theta);
}

Eigen::VectorXcd Ylm_vectorized(int l, int m, const Eigen::VectorXd &theta,
                                const Eigen::VectorXd &phi) {
  if (theta.size() != phi.size()) {
    throw std::invalid_argument("theta and phi must have the same size");
  }
  Eigen::VectorXcd result(theta.size());
  for (int i = 0; i < theta.size(); ++i) {
    result[i] = Ylm(l, m, theta[i], phi[i]);
  }
  return result;
}

} // namespace mathutils