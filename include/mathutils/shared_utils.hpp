#pragma once

/**
 * @file shared_utils.hpp
 * @brief Shared functions and utilities
 */

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace mathutils {
/**
 * @brief Compute the sign (1,-1, or 0) of a value.
 *
 * @tparam T The type of the input value (must be comparable to T{0}).
 * @param x The input value.
 * @return The sign of the input value.
 */
template <typename T> inline int sign(T x) { return (x > T{0}) - (x < T{0}); }

/**
 * @brief Convert Cartesian coordinates (x, y, z) to spherical coordinates (r,
 * theta, phi).
 *
 * @param xyz The input Cartesian coordinates [...(x,y,z)...] (N, 3)
 * @return The spherical coordinates [...(r, theta, phi)...] (N, 3)
 */
inline Eigen::MatrixXd rthetaphi_from_xyz(const Eigen::MatrixXd &xyz) {
  if (xyz.cols() != 3) {
    throw std::invalid_argument("xyz must have 3 columns");
  }
  int num_points = xyz.rows();
  Eigen::MatrixXd rthetaphi(num_points, 3);
  rthetaphi.setZero();
  for (int i = 0; i < num_points; ++i) {
    double x = xyz(i, 0);
    double y = xyz(i, 1);
    double z = xyz(i, 2);
    double r = std::sqrt(x * x + y * y + z * z);
    double rho = std::sqrt(x * x + y * y);
    double theta = std::atan2(rho, z);
    double phi = std::atan2(y, x);
    rthetaphi(i, 0) = r;
    rthetaphi(i, 1) = theta;
    rthetaphi(i, 2) = phi;
  }
  return rthetaphi;
}

/**
 * @brief Convert spherical coordinates (r, theta, phi) to Cartesian coordinates
 * (x, y, z).
 *
 * @param rthetaphi The input spherical coordinates [...(r, theta, phi)...] (N,
 * 3)
 * @return The Cartesian coordinates [...(x, y, z)...] (N, 3)
 */
inline Eigen::MatrixXd xyz_from_rthetaphi(const Eigen::MatrixXd &rthetaphi) {
  if (rthetaphi.cols() != 3) {
    throw std::invalid_argument("rthetaphi must have 3 columns");
  }
  int num_points = rthetaphi.rows();
  Eigen::MatrixXd xyz(num_points, 3);
  xyz.setZero();
  for (int i = 0; i < num_points; ++i) {
    double r = rthetaphi(i, 0);
    double theta = rthetaphi(i, 1);
    double phi = rthetaphi(i, 2);
    xyz(i, 0) = r * std::sin(theta) * std::cos(phi);
    xyz(i, 1) = r * std::sin(theta) * std::sin(phi);
    xyz(i, 2) = r * std::cos(theta);
  }
  return xyz;
}

/**
 * @brief Convert Cartesian coordinates (x, y, z) to unit sphere coordinates
 * (theta, phi) by projecting onto the unit sphere.
 *
 * @param xyz The input Cartesian coordinates [...(x,y,z)...] (N, 3)
 * @return The spherical coordinates [...(theta, phi)...] (N, 2)
 */
inline Eigen::MatrixXd thetaphi_from_xyz(const Eigen::MatrixXd &xyz) {
  if (xyz.cols() != 3) {
    throw std::invalid_argument("xyz must have 3 columns");
  }
  int num_points = xyz.rows();
  Eigen::MatrixXd thetaphi(num_points, 2);
  thetaphi.setZero();
  for (int i = 0; i < num_points; ++i) {
    double x = xyz(i, 0);
    double y = xyz(i, 1);
    double z = xyz(i, 2);
    double rho = std::sqrt(x * x + y * y);
    double theta = std::atan2(rho, z);
    double phi = std::atan2(y, x);
    thetaphi(i, 0) = theta;
    thetaphi(i, 1) = phi;
  }
  return thetaphi;
}
} // namespace mathutils