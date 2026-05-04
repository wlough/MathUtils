#pragma once

/**
 * @file spherical_harmonics.cpp
 * @brief Spherical harmonics
 */

#include "mathutils/special/spherical_harmonics.hpp"
// #include "mathutils/shared_utils.hpp"

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

Matrix<double> real_Ylm(int l, int m, const Matrix<double> &thetaphi_coord_P) {
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
  Matrix<double> result(thetaphi_coord_P.rows());
  for (int i = 0; i < thetaphi_coord_P.rows(); ++i) {
    double theta = thetaphi_coord_P(i, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(i, 1);
    result[i] = recursive_real_Ylm(l, m, theta, phi);
  }
  return result;
}

Matrix<std::complex<double>> Ylm(int l, int m,
                                 const Matrix<double> &thetaphi_coord_P) {

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
  Matrix<std::complex<double>> result(thetaphi_coord_P.rows());
  for (int i = 0; i < thetaphi_coord_P.rows(); ++i) {
    double theta = thetaphi_coord_P(i, 0);
    if (theta < 0 || theta > M_PI) {
      throw std::out_of_range("theta must be in the range [0, pi]");
    }
    double phi = thetaphi_coord_P(i, 1);
    result[i] = recursive_Ylm(l, m, theta, phi);
  }
  return result;
}

Matrix<double> compute_all_real_Ylm(int l_max,
                                    const Matrix<double> &thetaphi_coord_P) {
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
  Matrix<double> Y(num_points, num_modes);

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

Matrix<std::complex<double>>
compute_all_Ylm(int l_max, const Matrix<double> &thetaphi_coord_P) {
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
  Matrix<std::complex<double>> Y(num_points, num_modes);
  // Y.setZero();

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
} // namespace special
} // namespace mathutils
