/**
 * @file findiff.cpp
 * @brief Implementation of finite difference utilities
 */

#include "mathutils/findiff.hpp"
#include <algorithm>
#include <cmath>

namespace mathutils {
namespace findiff {

Eigen::MatrixXd fornberg_weights(const Eigen::VectorXd &x_stencil, double x0,
                                 int d_max) {
  if (d_max < 0) {
    throw std::invalid_argument("d_max must be non-negative");
  }

  const int n = x_stencil.size();
  if (n == 0) {
    throw std::invalid_argument("x_stencil cannot be empty");
  }

  // Initialize weights matrix: W(d, j) for derivative d at point j
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(d_max + 1, n);

  double c1 = 1.0;
  double c4 = x_stencil[0] - x0;
  W(0, 0) = 1.0;

  for (int i = 1; i < n; ++i) {
    const int mn = std::min(i, d_max);
    double c2 = 1.0;
    const double c5 = c4;
    c4 = x_stencil[i] - x0;

    for (int j = 0; j < i; ++j) {
      const double c3 = x_stencil[i] - x_stencil[j];
      c2 *= c3;

      if (j == i - 1) {
        // Update weights for the new point i
        for (int k = mn; k >= 1; --k) {
          W(k, i) = (c1 * (k * W(k - 1, i - 1) - c5 * W(k, i - 1))) / c2;
        }
        W(0, i) = (-c1 * c5 * W(0, i - 1)) / c2;
      }

      // Update weights for existing points
      for (int k = mn; k >= 1; --k) {
        W(k, j) = (c4 * W(k, j) - k * W(k - 1, j)) / c3;
      }
      W(0, j) = (c4 * W(0, j)) / c3;
    }
    c1 = c2;
  }

  return W;
}

Eigen::MatrixXd fornberg_weights(const std::vector<double> &x_stencil,
                                 double x0, int d_max) {
  if (x_stencil.empty()) {
    throw std::invalid_argument("x_stencil cannot be empty");
  }

  // Convert std::vector to Eigen::VectorXd
  Eigen::VectorXd x_eigen(x_stencil.size());
  for (size_t i = 0; i < x_stencil.size(); ++i) {
    x_eigen[i] = x_stencil[i];
  }

  return fornberg_weights(x_eigen, x0, d_max);
}

} // namespace findiff
} // namespace mathutils