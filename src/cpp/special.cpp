#include "mathutils/special/special.hpp"
#include <cmath>
#include <complex>

namespace mathutils {
namespace special {

mathutils::SimpleGenerator<double>
generate_reduced_spherical_Plm_recursion_three_term_upward_ell(int l_start,
                                                               int l_end,
                                                               double theta) {
  if (l_start < 0 || l_end < l_start) {
    co_return;
  }
  int m = l_start;
  double P_ell_minus_one_m = reduced_spherical_Pmm(m, theta);
  // ell = l_start
  co_yield P_ell_minus_one_m;
  if (l_start == l_end) {
    co_return;
  }
  double cos_theta = std::cos(theta);
  double P_ell_m = cos_theta * std::sqrt(2 * m + 3) * P_ell_minus_one_m;
  // ell = l_start + 1
  co_yield P_ell_m;
  for (int ell = m + 2; ell <= l_end; ++ell) {
    const double den_ell_m = (ell - m) * 1.0 * (ell + m);
    const double c_ell_m =
        std::sqrt(((2.0 * ell - 1.0) * (2.0 * ell + 1.0)) / den_ell_m);
    const double den_ell_minus_one_m =
        (ell - m) * 1.0 * (ell + m) * (2.0 * ell - 3.0);
    const double c_ell_minus_one_m =
        std::sqrt(((ell - m - 1.0) * (ell + m - 1.0) * (2.0 * ell + 1.0)) /
                  den_ell_minus_one_m);
    const double q =
        cos_theta * c_ell_m * P_ell_m - c_ell_minus_one_m * P_ell_minus_one_m;
    P_ell_minus_one_m = P_ell_m;
    P_ell_m = q;
    co_yield P_ell_m;
  }
}

mathutils::SimpleGenerator<std::pair<int, double>>
enumerate_reduced_spherical_Plm_recursion_three_term_upward_ell(int l_start,
                                                                int l_end,
                                                                double theta) {
  if (l_start < 0 || l_end < l_start) {
    co_return;
  }
  int m = l_start;
  double P_ell_minus_one_m = reduced_spherical_Pmm(m, theta);
  // ell = l_start
  co_yield {l_start, P_ell_minus_one_m};
  if (l_start == l_end) {
    co_return;
  }
  double cos_theta = std::cos(theta);
  double P_ell_m = cos_theta * std::sqrt(2 * m + 3) * P_ell_minus_one_m;
  // ell = l_start + 1
  co_yield {l_start + 1, P_ell_m};
  for (int ell = m + 2; ell <= l_end; ++ell) {
    const double den_ell_m = (ell - m) * 1.0 * (ell + m);
    const double c_ell_m =
        std::sqrt(((2.0 * ell - 1.0) * (2.0 * ell + 1.0)) / den_ell_m);
    const double den_ell_minus_one_m =
        (ell - m) * 1.0 * (ell + m) * (2.0 * ell - 3.0);
    const double c_ell_minus_one_m =
        std::sqrt(((ell - m - 1.0) * (ell + m - 1.0) * (2.0 * ell + 1.0)) /
                  den_ell_minus_one_m);
    const double q =
        cos_theta * c_ell_m * P_ell_m - c_ell_minus_one_m * P_ell_minus_one_m;
    P_ell_minus_one_m = P_ell_m;
    P_ell_m = q;
    co_yield {ell, P_ell_m};
  }
}

Eigen::VectorXd real_Ylm(int l, int m,
                         const Eigen::MatrixXd &thetaphi_coord_P) {
  // check l and m
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  // check thetaphi_coord_P
  if (thetaphi_coord_P.cols() != 2) {
    throw std::invalid_argument("thetaphi_coord_P must have 2 columns");
  }
  Eigen::VectorXd result(thetaphi_coord_P.rows());
  for (int i = 0; i < thetaphi_coord_P.rows(); ++i) {
    double theta = thetaphi_coord_P(i, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(i, 1);
    result(i) = recursive_real_Ylm(l, m, theta, phi);
  }
  return result;
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

  for (int p{0}; p < num_points; ++p) {
    double theta = thetaphi_coord_P(p, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(p, 1);
    double reduced_theta =
        std::min(theta, M_PI - theta); // transform theta to [0, pi/2]
    // m = 0
    for (auto [l, reduced_P] :
         enumerate_reduced_spherical_Plm_recursion_three_term_upward_ell(
             0, l_max, reduced_theta)) {
      int n = l * (l + 1);
      Y(p, n) = ((theta > M_PI_2) && (l & 1)) ? -reduced_P : reduced_P;
    }
    // m = +/-1, ... , +/-l_max
    for (int m = 1; m <= l_max; ++m) {
      for (auto [l, reduced_P] :
           enumerate_reduced_spherical_Plm_recursion_three_term_upward_ell(
               m, l_max, reduced_theta)) {
        int n_plus = l * (l + 1) + m;
        int n_minus = l * (l + 1) - m;
        // Q = (-1)^m * sqrt(2) * P_{l|m|}(theta)
        double Q = ((m & 1) ^ ((theta > M_PI_2) && ((l - m) & 1)))
                       ? -std::sqrt(2) * reduced_P
                       : std::sqrt(2) * reduced_P;
        // Y_{l |m|} = (-1)^m * sqrt(2) * P_{l|m|}*cos(|m|*phi)
        Y(p, n_plus) = std::cos(m * phi) * Q;
        // Y_{l, -|m|} = (-1)^m * sqrt(2) * P_{l|m|}*sin(|m|*phi)
        Y(p, n_minus) = std::sin(m * phi) * Q;
      }
    }
  } // end for p
  return Y;
}

Eigen::VectorXcd Ylm(int l, int m, const Eigen::MatrixXd &thetaphi_coord_P) {

  // check l and m
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  // check thetaphi_coord_P
  if (thetaphi_coord_P.cols() != 2) {
    throw std::invalid_argument("thetaphi_coord_P must have 2 columns");
  }
  Eigen::VectorXcd result(thetaphi_coord_P.rows());
  for (int i = 0; i < thetaphi_coord_P.rows(); ++i) {
    double theta = thetaphi_coord_P(i, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(i, 1);
    result(i) = recursive_Ylm(l, m, theta, phi);
  }
  return result;
}

Eigen::MatrixXcd compute_all_Ylm(int l_max,
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
  Eigen::MatrixXcd Y(num_points, num_modes);
  Y.setZero();

  for (int p{0}; p < num_points; ++p) {
    double theta = thetaphi_coord_P(p, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(p, 1);
    double reduced_theta =
        std::min(theta, M_PI - theta); // transform theta to [0, pi/2]
    // m = 0
    for (auto [l, reduced_P] :
         enumerate_reduced_spherical_Plm_recursion_three_term_upward_ell(
             0, l_max, reduced_theta)) {
      int n = l * (l + 1);
      Y(p, n) = ((theta > M_PI_2) && (l & 1)) ? -reduced_P : reduced_P;
    }
    // m = +/-1, ... , +/-l_max
    for (int m = 1; m <= l_max; ++m) {
      for (auto [l, reduced_P] :
           enumerate_reduced_spherical_Plm_recursion_three_term_upward_ell(
               m, l_max, reduced_theta)) {
        int n_plus = l * (l + 1) + m;
        int n_minus = l * (l + 1) - m;
        // Y_{l |m|} = P_{l|m|}*exp(i|m|phi)
        Y(p, n_plus) =
            ((theta > M_PI_2) && ((l - m) & 1))
                ? -reduced_P * std::exp(std::complex<double>(0, m * phi))
                : reduced_P * std::exp(std::complex<double>(0, m * phi));
        // Y_{l, -|m|} = (-1)^m conj(Y_{l|m|})
        Y(p, n_minus) =
            (m & 1) ? -std::conj(Y(p, n_plus)) : std::conj(Y(p, n_plus));
      }
    }
  } // end for p
  return Y;
}

//////////////////////////////////////////////////////
// Alternative implementations of spherical harmonics.
// May be unstable for large l and m.

double series_real_Ylm(int l, int m, double theta, double phi) {
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta < 0 || theta > M_PI) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  int abs_m = std::abs(m);
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);
  auto [log_abs_cos_theta, arg_over_pi_cos_theta] =
      ReLogRe_ImLogRe_over_pi(cos_theta);
  auto [log_abs_sin_theta, arg_over_pi_sin_theta] =
      ReLogRe_ImLogRe_over_pi(sin_theta);
  double Xlm = 0.0;
  for (int k{0}; k <= (l - abs_m) / 2; k++) {
    int ImLog_over_pi_Xklm =
        k + (l - abs_m) * arg_over_pi_cos_theta + abs_m * arg_over_pi_sin_theta;
    double ReLog_Xklm =
        -0.5 * std::log(4 * M_PI) - (abs_m + 2 * k) * std::log(2) +
        0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) +
        0.5 * log_factorial(l - abs_m) - log_factorial(l - abs_m - 2 * k) -
        log_factorial(abs_m + k) - log_factorial(k);
    //  +
    // (l - abs_m - 2 * k) * log_abs_cos_theta +
    // (abs_m + 2 * k) * log_abs_sin_theta;
    int pow_cos = (l - abs_m - 2 * k);
    int pow_sin = (abs_m + 2 * k);
    ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
    ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
    Xlm += minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
  }
  return (m < 0)   ? std::sqrt(2) * std::sin(abs_m * phi) * Xlm
         : (m > 0) ? std::sqrt(2) * std::cos(abs_m * phi) * Xlm
                   : Xlm;
}

Eigen::VectorXd series_real_Ylm(int l, int m,
                                const Eigen::MatrixXd &thetaphi_coord_P) {

  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (thetaphi_coord_P.cols() != 2) {
    throw std::invalid_argument("thetaphi_coord_P must have 2 columns");
  }
  int abs_m = std::abs(m);
  int num_points = thetaphi_coord_P.rows();
  Eigen::VectorXd Y(num_points);
  Y.setZero();
  for (int p = 0; p < num_points; ++p) {
    double theta = thetaphi_coord_P(p, 0);
    double phi = thetaphi_coord_P(p, 1);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    auto [log_abs_cos_theta, arg_over_pi_cos_theta] =
        ReLogRe_ImLogRe_over_pi(cos_theta);
    auto [log_abs_sin_theta, arg_over_pi_sin_theta] =
        ReLogRe_ImLogRe_over_pi(sin_theta);
    double Xlm = 0.0;
    for (int k{0}; k <= (l - abs_m) / 2; k++) {
      int ImLog_over_pi_Xklm = k + (l - abs_m) * arg_over_pi_cos_theta +
                               abs_m * arg_over_pi_sin_theta;
      double ReLog_Xklm =
          -0.5 * std::log(4 * M_PI) - (abs_m + 2 * k) * std::log(2) +
          0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) +
          0.5 * log_factorial(l - abs_m) - log_factorial(l - abs_m - 2 * k) -
          log_factorial(abs_m + k) - log_factorial(k);
      // +
      // (l - abs_m - 2 * k) * log_abs_cos_theta +
      // (abs_m + 2 * k) * log_abs_sin_theta;
      int pow_cos = (l - abs_m - 2 * k);
      int pow_sin = (abs_m + 2 * k);
      ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
      ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
      Xlm += minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
    }
    Y(p) = (m < 0)   ? std::sqrt(2) * std::sin(abs_m * phi) * Xlm
           : (m > 0) ? std::sqrt(2) * std::cos(abs_m * phi) * Xlm
                     : Xlm; // m == 0
  }
  return Y;
}

std::complex<double> series_Ylm(int l, int m, double theta, double phi) {
  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (theta < 0 || theta > M_PI) {
    throw std::out_of_range("theta must be in the range [0, pi]");
  }
  int abs_m = std::abs(m);
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);
  auto [log_abs_cos_theta, arg_over_pi_cos_theta] =
      ReLogRe_ImLogRe_over_pi(cos_theta);
  auto [log_abs_sin_theta, arg_over_pi_sin_theta] =
      ReLogRe_ImLogRe_over_pi(sin_theta);
  double Xlm = 0.0;
  for (int k{0}; k <= (l - abs_m) / 2; k++) {
    int ImLog_over_pi_Xklm =
        k + (l - abs_m) * arg_over_pi_cos_theta + abs_m * arg_over_pi_sin_theta;
    double ReLog_Xklm = -0.5 * std::log(4 * M_PI) - abs_m * std::log(2) +
                        0.5 * std::log(2 * l + 1) +
                        0.5 * log_factorial(l + abs_m) +
                        0.5 * log_factorial(l - abs_m) - 2 * k * std::log(2) -
                        log_factorial(l - abs_m - 2 * k) -
                        log_factorial(abs_m + k) - log_factorial(k);
    int pow_cos = (l - abs_m - 2 * k);
    int pow_sin = (abs_m + 2 * k);
    ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
    ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
    Xlm += minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
  }
  return (m < 0) ? std::exp(std::complex<double>(0, m * phi)) * Xlm
         : (m > 0)
             ? static_cast<std::complex<double>>(minus_one_to_int_pow(m)) *
                   std::exp(std::complex<double>(0, m * phi)) * Xlm
             : static_cast<std::complex<double>>(Xlm);
}

Eigen::VectorXcd series_Ylm(int l, int m,
                            const Eigen::MatrixXd &thetaphi_coord_P) {

  if (l < 0) {
    throw std::out_of_range("l must be non-negative");
  }
  if (m < -l || m > l) {
    throw std::out_of_range("m must be in the range [-l, l]");
  }
  if (thetaphi_coord_P.cols() != 2) {
    throw std::invalid_argument("thetaphi_coord_P must have 2 columns");
  }
  int abs_m = std::abs(m);
  int num_points = thetaphi_coord_P.rows();
  Eigen::VectorXcd Y(num_points);
  Y.setZero();
  for (int p = 0; p < num_points; ++p) {
    double theta = thetaphi_coord_P(p, 0);
    double phi = thetaphi_coord_P(p, 1);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    auto [log_abs_cos_theta, arg_over_pi_cos_theta] =
        ReLogRe_ImLogRe_over_pi(cos_theta);
    auto [log_abs_sin_theta, arg_over_pi_sin_theta] =
        ReLogRe_ImLogRe_over_pi(sin_theta);
    for (int k{0}; k <= (l - abs_m) / 2; k++) {
      // adding (m + abs_m) / 2 here takes care of (-1)^m for m>0
      int ImLog_over_pi_Xklm = (m + abs_m) / 2 + k +
                               (l - abs_m) * arg_over_pi_cos_theta +
                               abs_m * arg_over_pi_sin_theta;
      double ReLog_Xklm =
          -0.5 * std::log(4 * M_PI) - (abs_m + 2 * k) * std::log(2) +
          0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m) +
          0.5 * log_factorial(l - abs_m) - log_factorial(l - abs_m - 2 * k) -
          log_factorial(abs_m + k) - log_factorial(k);
      //  +
      // (l - abs_m - 2 * k) * log_abs_cos_theta +
      // (abs_m + 2 * k) * log_abs_sin_theta;
      int pow_cos = (l - abs_m - 2 * k);
      int pow_sin = (abs_m + 2 * k);
      ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
      ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
      Y(p) += minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
    }
    Y(p) *= std::exp(std::complex<double>(0, m * phi));
  }
  return Y;
}

Eigen::MatrixXd
compute_all_series_real_Ylm(int l_max,
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
                            log_factorial(k) - log_factorial(k);
        //  +
        // (l - 2 * k) * log_abs_cos_theta +
        // (2 * k) * log_abs_sin_theta;
        int pow_cos = l - 2 * k;
        int pow_sin = 2 * k;
        ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
        ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
        Y(p, n) +=
            minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
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
              log_factorial(m + k) - log_factorial(k);
          // +
          // (l - m - 2 * k) * log_abs_cos_theta +
          // (m + 2 * k) * log_abs_sin_theta;
          int pow_cos = (l - m - 2 * k);
          int pow_sin = (m + 2 * k);
          ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
          ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
          Y(p, n_plus) +=
              minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
        }
        Y(p, n_minus) = std::sqrt(2) * std::sin(m * phi) * Y(p, n_plus);
        Y(p, n_plus) *= std::sqrt(2) * std::cos(m * phi);
      }
    }
  }
  return Y;
}

Eigen::MatrixXcd
compute_all_series_Ylm(int l_max, const Eigen::MatrixXd &thetaphi_coord_P) {
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
  Eigen::MatrixXcd Y(num_points, num_modes);
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
                            log_factorial(k) - log_factorial(k);
        //  +
        // (l - 2 * k) * log_abs_cos_theta +
        // (2 * k) * log_abs_sin_theta;
        int pow_cos = l - 2 * k;
        int pow_sin = 2 * k;
        ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
        ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
        Y(p, n) +=
            minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
      }
      // m=-l,...,l
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
              log_factorial(m + k) - log_factorial(k);
          //  +
          // (l - m - 2 * k) * log_abs_cos_theta +
          // (m + 2 * k) * log_abs_sin_theta;
          int pow_cos = (l - m - 2 * k);
          int pow_sin = (m + 2 * k);
          ReLog_Xklm += (pow_cos > 0) ? (pow_cos * log_abs_cos_theta) : 0.0;
          ReLog_Xklm += (pow_sin > 0) ? (pow_sin * log_abs_sin_theta) : 0.0;
          Y(p, n_plus) +=
              minus_one_to_int_pow(ImLog_over_pi_Xklm) * std::exp(ReLog_Xklm);
        }
        Y(p, n_minus) =
            Y(p, n_plus) * std::exp(std::complex<double>(0, -m * phi));
        Y(p, n_plus) *=
            static_cast<std::complex<double>>(minus_one_to_int_pow(m)) *
            std::exp(std::complex<double>(0, m * phi));
      }
    }
  }
  return Y;
}

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

Eigen::MatrixXd
old_compute_all_real_Ylm(int l_max, const Eigen::MatrixXd &thetaphi_coord_P) {
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
        Y(p, n) = old_real_Ylm(l, m, theta, phi);
      }
    }
  }
  return Y;
}

} // namespace special
} // namespace mathutils

////////////////////////////////
// Spherical harmonic fitting //
////////////////////////////////
namespace mathutils {
namespace special {

Eigen::MatrixXd fit_real_sh_coefficients_to_points(const Eigen::MatrixXd &XYZ0,
                                                   int l_max,
                                                   double reg_lambda) {
  // ---------------------------------------------------------------------
  // 0. Basic argument checks
  // ---------------------------------------------------------------------
  if (l_max < 0)
    throw std::invalid_argument(
        "fit_real_sh_coefficients_to_points: l_max must be non-negative.");

  if (XYZ0.cols() != 3)
    throw std::invalid_argument("fit_real_sh_coefficients_to_points: XYZ0 must "
                                "have exactly 3 columns.");

  if (XYZ0.rows() == 0)
    throw std::invalid_argument("fit_real_sh_coefficients_to_points: XYZ0 must "
                                "contain at least one point.");

  // ---------------------------------------------------------------------
  // 1. Dimensions
  // ---------------------------------------------------------------------
  const int num_modes = (l_max + 1) * (l_max + 1); // (l_max+1)^2
  // const int N = static_cast<int>(XYZ0.rows());

  // ---------------------------------------------------------------------
  // 2. Build the design matrix  Y  (N × num_modes)
  //    Only directions (θ,φ) matter; radius is ignored.
  // ---------------------------------------------------------------------
  Eigen::MatrixXd thetaPhi = mathutils::thetaphi_from_xyz(XYZ0); // (N,2)
  Eigen::MatrixXd Y = compute_all_real_Ylm(l_max, thetaPhi);

  // ---------------------------------------------------------------------
  // 3. Construct Laplace–Beltrami weights  ℓ(ℓ+1)
  // ---------------------------------------------------------------------
  Eigen::VectorXd laplace(num_modes);
  for (int n = 0; n < num_modes; ++n) {
    auto [l, m] = spherical_harmonic_index_lm_N(n);
    laplace(n) = static_cast<double>(l * (l + 1));
  }

  // ---------------------------------------------------------------------
  // 4. Assemble the normal matrix  M = YᵀY  (SPD)  and add regularisation
  //    directly to its diagonal:  M_ii += λ (ℓ(ℓ+1))²
  // ---------------------------------------------------------------------
  Eigen::MatrixXd M = Y.transpose() * Y; // (num_modes, num_modes)
  if (reg_lambda > 0.0)
    M.diagonal().array() += reg_lambda * laplace.array().square();

  // ---------------------------------------------------------------------
  // 5. Right-hand side  B = Yᵀ X
  // ---------------------------------------------------------------------
  Eigen::MatrixXd B = Y.transpose() * XYZ0; // (num_modes, 3)

  // ---------------------------------------------------------------------
  // 6. Solve  M A = B   via Cholesky (faster & stabler than explicit inverse)
  // ---------------------------------------------------------------------
  Eigen::LLT<Eigen::MatrixXd> chol(M);
  if (chol.info() != Eigen::Success)
    throw std::runtime_error("fit_real_sh_coefficients_to_points: normal "
                             "matrix not SPD (ill-posed fit).");

  Eigen::MatrixXd A = chol.solve(B); // coefficients
  return A;                          // (num_modes, 3)
}
} // namespace special
} // namespace mathutils