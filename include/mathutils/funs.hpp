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

inline double Heaviside(double x) {
  if (x < 0.0) {
    return 0.0;
  } else if (x > 0.0) {
    return 1.0;
  } else {
    return 0.5; // H(0) = 1/2
  }
}

inline int Heaviside(int x) {
  if (x < 0) {
    return 0;
  } else {
    return 1;
  }
}

inline int sign(int x) { return (x > 0) - (x < 0); }
inline int sign(double x) { return (x > 0.0) - (x < 0.0); }

inline int spherical_harmonic_index_n_LM(int l, int m) {
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  return l * (l + 1) + m;
  // return l*l+l+l = l**2+2l = l
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

inline double magnitude_Ylm(int l, int abs_m, double abs_cos_theta) {
  double epsilon = 1e-8;
  double minus_log4 = std::log(0.25);

  int ell_minus_m = l - abs_m;
  double abs_sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);

  double rlm = 1.0 / (2.0 * std::sqrt(M_PI));
  int k_bound = (ell_minus_m) / 2;
  int sk = 1;
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
    double abs_tan_theta = abs_sin_theta / abs_cos_theta;
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

inline std::complex<double> Ylm(int l, int m, double theta, double phi) {
  if (theta > M_PI || theta < 0.0) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  return std::exp(std::complex<double>(0, m * phi)) *
         phi_independent_Ylm(l, m, theta);
  // if (theta > M_PI || theta < 0.0) {
  //   throw std::out_of_range("theta must be in the range [0, pi]");
  // }
  // int abs_m = std::abs(m);
  // double cos_theta = std::cos(theta);
  // double abs_cos_theta = std::abs(cos_theta);
  // std::complex<double> val = std::exp(std::complex<double>(0, m * phi)) *
  //                            magnitude_Ylm(l, abs_m, abs_cos_theta);
  // if ((m > 0) && (abs_m % 2 == 1)) {
  //   val *= -1;
  // }
  // if ((cos_theta < 0) && (l - abs_m % 2 == 1)) {
  //   val *= -1;
  // }

  // return val;
}

Eigen::VectorXcd Ylm_vectorized(int l, int m, const Eigen::VectorXd &theta,
                                const Eigen::VectorXd &phi) {
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta.size() != phi.size()) {
    throw std::invalid_argument("theta and phi must have the same size");
  }
  Eigen::VectorXcd result(theta.size());
  for (int i = 0; i < theta.size(); ++i) {
    result[i] = Ylm(l, m, theta[i], phi[i]);
  }
  return result;
}

// inline double real_Ylm(int l, int m, double theta, double phi) {

//   if (theta > M_PI || theta < 0.0) {
//     throw std::out_of_range("theta must be in the range [0, pi]");
//   }
//   int abs_m = std::abs(m);
//   double cos_theta = std::cos(theta);
//   double abs_cos_theta = std::abs(cos_theta);
//   double val = magnitude_Ylm(l, abs_m, abs_cos_theta);
//   int s = 1;

//   if (m < 0) {
//     val *= std::sqrt(2) * std::sin(abs_m * phi);
//     if ((cos_theta < 0) && (l - abs_m % 2 == 1)) {
//       s *= -1;
//     }
//     if (abs_m % 2 == 1) {
//       s *= -1;
//     }
//     return s * val;
//   }
//   if (m == 0) {
//     if ((cos_theta < 0) && (l % 2 == 1)) {
//       s *= -1;
//     }
//     return s * val;
//   }
//   val *= std::sqrt(2) * std::cos(abs_m * phi);
//   if ((cos_theta < 0) && (l - abs_m % 2 == 1)) {
//     s *= -1;
//   }
//   return s * val;
// }

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

Eigen::VectorXd real_Ylm_vectorized(int l, int m, const Eigen::VectorXd &theta,
                                    const Eigen::VectorXd &phi) {
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta.size() != phi.size()) {
    throw std::invalid_argument("theta and phi must have the same size");
  }
  Eigen::VectorXd result(theta.size());
  for (int i = 0; i < theta.size(); ++i) {
    result[i] = real_Ylm(l, m, theta[i], phi[i]);
  }
  return result;
}

inline int spherical_harmonic_s_lmtheta(int l, int m, double theta) {
  int sign_m = (m > 0) - (m < 0);
  int sign_cos_theta = (theta < M_PI_2) - (theta > M_PI_2);
  int s = 1;
  if (l % 2 == 1) {
    s *= sign_cos_theta;
  }
  if (m % 2 == 1) {
    s *= -sign_m * sign_cos_theta;
  }
  return s;
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

  return Y;
}

} // namespace mathutils
