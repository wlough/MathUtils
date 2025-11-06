/**
 * @file findiff.cpp
 * @brief Implementation of finite difference utilities
 */

#include "mathutils/findiff.hpp"
#include <Eigen/Dense> // Eigen::MatrixXd, Eigen::VectorXd, Zero()
#include <algorithm>   // std::min
#include <cmath>
#include <cstddef>   // size_t
#include <stdexcept> // std::invalid_argument
#include <vector>    // std::vector

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

namespace {
// wrap a difference into (-T/2, T/2]
inline double wrap_diff_half_period(double d, double T) {
  double y = std::fmod(d + 0.5 * T, T);
  if (y < 0.0)
    y += T;
  return y - 0.5 * T;
}

// builds stencil based on derivative and order of accuracy
static std::vector<int> build_stencil(int deriv_order, int acc,
                                      bool symmetric = false) {
  int stencil_width = deriv_order + acc;
  if (symmetric && (deriv_order % 2 == 0)) {
    stencil_width -= 1;
  }
  if (stencil_width % 2 == 0) {
    stencil_width += 1;
  }
  int stencil_radius = (stencil_width - 1) / 2;
  std::vector<int> stencil(stencil_width);
  for (int i = 0; i < stencil_width; ++i) {
    stencil[i] = i - stencil_radius;
  }
  return stencil;
}

// Fornberg in-place recursion for a single row; returns weights for deriv_order
static void fornberg_row_weights(const std::vector<double> &dx, int deriv_order,
                                 std::vector<double> &w_out) {
  const int sw = static_cast<int>(dx.size());
  const int dmax = deriv_order;
  std::vector<double> W((dmax + 1) * sw, 0.0);
  auto W_at = [&](int k, int i) -> double & { return W[k * sw + i]; };

  double c1 = 1.0;
  double c4 = dx[0];
  W_at(0, 0) = 1.0;

  for (int i = 1; i < sw; ++i) {
    const int mn = std::min(i, dmax);
    double c2 = 1.0, c5 = c4;
    c4 = dx[i];
    for (int j = 0; j <= i - 1; ++j) {
      const double c3 = dx[i] - dx[j];
      c2 *= c3;
      if (j == i - 1) {
        for (int k = mn; k >= 1; --k) {
          W_at(k, i) =
              (c1 * (k * W_at(k - 1, i - 1) - c5 * W_at(k, i - 1))) / c2;
        }
        W_at(0, i) = (-c1 * c5 * W_at(0, i - 1)) / c2;
      }
      for (int k = mn; k >= 1; --k) {
        W_at(k, j) = (c4 * W_at(k, j) - k * W_at(k - 1, j)) / c3;
      }
      W_at(0, j) = (c4 * W_at(0, j)) / c3;
    }
    c1 = c2;
  }

  w_out.resize(sw);
  for (int i = 0; i < sw; ++i)
    w_out[i] = W[dmax * sw + i];
}

} // anonymous namespace

namespace mathutils {
namespace findiff {

// for periodic boundary conditions
void FiniteDifference1D::build(const BuildSpecPeriodic &spec) {
  const auto &x = spec.x;
  const auto &stencil = spec.stencil;
  const int deriv_order = spec.deriv_order;
  const int Nx = static_cast<int>(x.size());
  const double period = spec.period;

  if (Nx <= 0)
    throw std::invalid_argument("x must be nonempty");
  if (stencil.empty())
    throw std::invalid_argument("stencil must be nonempty");
  if (deriv_order < 0)
    throw std::invalid_argument("deriv_order must be >= 0");

  auto [min_it, max_it] = std::minmax_element(stencil.begin(), stencil.end());
  const int min_offset = *min_it, max_offset = *max_it;

  const int interior_start = std::max(0, -min_offset);
  const int interior_end = std::min(Nx, Nx - max_offset);

  const int b0_start = 0, b0_end = interior_start;
  const int b1_start = interior_end, b1_end = Nx;

  // boundary stencils
  std::vector<int> b0_sten, b1_sten;
  b0_sten = stencil;
  b1_sten = stencil;

  std::vector<Eigen::Triplet<double, Index64>> trips;
  trips.reserve(static_cast<size_t>(Nx) * stencil.size());

  auto get_indices = [&](int n, const std::vector<int> &off) {
    std::vector<Index64> idx(off.size());
    for (size_t k = 0; k < off.size(); ++k) {
      int t = n + off[k];
      int m = t % Nx;
      if (m < 0)
        m += Nx;
      idx[k] = static_cast<Index64>(m);
    }
    return idx;
  };

  auto get_dx = [&](const std::vector<Index64> &ind, int n) {
    std::vector<double> dx(ind.size());
    const double T = spec.period;
    for (size_t i = 0; i < ind.size(); ++i)
      dx[i] = wrap_diff_half_period(x[static_cast<int>(ind[i])] - x[n], T);
    return dx;
  };

  std::vector<double> wrow;

  // boundary 0 rows
  for (int n = b0_start; n < b0_end; ++n) {
    auto Kn = get_indices(n, b0_sten);
    auto dx = get_dx(Kn, n);
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }
  // interior rows
  for (int n = interior_start; n < interior_end; ++n) {
    std::vector<Index64> Kn(stencil.size());
    for (size_t i = 0; i < stencil.size(); ++i)
      Kn[i] = static_cast<Index64>(n + stencil[i]);
    auto dx = get_dx(Kn, n);
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }
  // boundary 1 rows
  for (int n = b1_start; n < b1_end; ++n) {
    auto Kn = get_indices(n, b1_sten);
    auto dx = get_dx(Kn, n);
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }

  Nx_ = Nx;
  D_.resize(Nx_, Nx_);
  D_.setFromTriplets(trips.begin(), trips.end());
  D_.makeCompressed(); // CSR
}

// for NON-periodic boundary conditions
void FiniteDifference1D::build(const BuildSpecNonPeriodic &spec) {
  const auto &x = spec.x;
  const auto &interior_stencil = spec.interior_stencil;
  const auto &boundary0_stencil = spec.boundary0_stencil;
  const auto &boundary1_stencil = spec.boundary1_stencil;
  const int deriv_order = spec.deriv_order;
  const int Nx = static_cast<int>(x.size());

  if (Nx <= 0)
    throw std::invalid_argument("x must be nonempty");
  if (interior_stencil.empty())
    throw std::invalid_argument("interior_stencil must be nonempty");
  if (deriv_order < 0)
    throw std::invalid_argument("deriv_order must be >= 0");

  auto [min_it, max_it] =
      std::minmax_element(interior_stencil.begin(), interior_stencil.end());
  const int min_offset = *min_it, max_offset = *max_it;

  const int interior_start = std::max(0, -min_offset);
  const int interior_end = std::min(Nx, Nx - max_offset);

  const int b0_start = 0, b0_end = interior_start;
  const int b1_start = interior_end, b1_end = Nx;

  std::vector<Eigen::Triplet<double, Index64>> trips;
  trips.reserve(static_cast<size_t>(Nx) * interior_stencil.size());

  std::vector<double> wrow;

  // boundary 0 rows
  std::vector<Index64> boundary0Kn(boundary0_stencil.size());
  for (size_t i = 0; i < boundary0_stencil.size(); ++i)
    boundary0Kn[i] = static_cast<Index64>(boundary0_stencil[i]);
  for (int n = b0_start; n < b0_end; ++n) {
    std::vector<Index64> Kn = boundary0Kn;
    std::vector<double> dx(Kn.size());
    for (size_t i = 0; i < Kn.size(); ++i)
      dx[i] = x[static_cast<int>(Kn[i])] - x[n];
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }
  // interior rows
  for (int n = interior_start; n < interior_end; ++n) {
    std::vector<Index64> Kn(interior_stencil.size());
    for (size_t i = 0; i < interior_stencil.size(); ++i)
      Kn[i] = static_cast<Index64>(n + interior_stencil[i]);
    std::vector<double> dx(Kn.size());
    for (size_t i = 0; i < Kn.size(); ++i)
      dx[i] = x[static_cast<int>(Kn[i])] - x[n];
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }
  // boundary 1 rows
  std::vector<Index64> boundary1Kn(boundary1_stencil.size());
  for (size_t i = 0; i < boundary1_stencil.size(); ++i)
    boundary1Kn[i] = static_cast<Index64>(Nx - 1 + boundary1_stencil[i]);
  for (int n = b1_start; n < b1_end; ++n) {
    std::vector<Index64> Kn = boundary1Kn;
    std::vector<double> dx(Kn.size());
    for (size_t i = 0; i < Kn.size(); ++i)
      dx[i] = x[static_cast<int>(Kn[i])] - x[n];
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }

  Nx_ = Nx;
  D_.resize(Nx_, Nx_);
  D_.setFromTriplets(trips.begin(), trips.end());
  D_.makeCompressed(); // CSR
}

void FiniteDifference1D::build(const BuildSpec &spec) {
  const auto &x = spec.x;
  const auto &stencil = spec.stencil;
  const int deriv_order = spec.deriv_order;
  const int Nx = static_cast<int>(x.size());

  if (Nx <= 0)
    throw std::invalid_argument("x must be nonempty");
  if (stencil.empty())
    throw std::invalid_argument("stencil must be nonempty");
  if (deriv_order < 0)
    throw std::invalid_argument("deriv_order must be >= 0");

  auto [min_it, max_it] = std::minmax_element(stencil.begin(), stencil.end());
  const int min_offset = *min_it, max_offset = *max_it;

  const int interior_start = std::max(0, -min_offset);
  const int interior_end = std::min(Nx, Nx - max_offset);

  const int b0_start = 0, b0_end = interior_start;
  const int b1_start = interior_end, b1_end = Nx;

  // boundary stencils
  std::vector<int> b0_sten, b1_sten;
  if (spec.period.has_value()) {
    b0_sten = stencil;
    b1_sten = stencil;
  } else {
    b0_sten = spec.boundary0_stencil.value_or(std::vector<int>{});
    b1_sten = spec.boundary1_stencil.value_or(std::vector<int>{});
    if (b0_sten.empty()) {
      b0_sten.resize(stencil.size());
      std::transform(stencil.begin(), stencil.end(), b0_sten.begin(),
                     [&](int a) { return a - min_offset; });
    }
    if (b1_sten.empty()) {
      b1_sten.resize(stencil.size());
      std::transform(stencil.begin(), stencil.end(), b1_sten.begin(),
                     [&](int a) { return a - max_offset; });
    }
  }

  std::vector<Eigen::Triplet<double, Index64>> trips;
  trips.reserve(static_cast<size_t>(Nx) * stencil.size());

  auto get_indices = [&](int n, const std::vector<int> &off) {
    std::vector<Index64> idx(off.size());
    if (spec.period.has_value()) {
      for (size_t k = 0; k < off.size(); ++k) {
        int t = n + off[k];
        int m = t % Nx;
        if (m < 0)
          m += Nx;
        idx[k] = static_cast<Index64>(m);
      }
    } else {
      for (size_t k = 0; k < off.size(); ++k)
        idx[k] = static_cast<Index64>(off[k]);
    }
    return idx;
  };

  auto get_dx = [&](const std::vector<Index64> &ind, int n) {
    std::vector<double> dx(ind.size());
    if (spec.period.has_value()) {
      const double T = *spec.period;
      for (size_t i = 0; i < ind.size(); ++i)
        dx[i] = wrap_diff_half_period(x[static_cast<int>(ind[i])] - x[n], T);
    } else {
      for (size_t i = 0; i < ind.size(); ++i)
        dx[i] = x[static_cast<int>(ind[i])] - x[n];
    }
    return dx;
  };

  std::vector<double> wrow;

  // boundary 0 rows
  for (int n = b0_start; n < b0_end; ++n) {
    auto Kn = get_indices(n, b0_sten);
    auto dx = get_dx(Kn, n);
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }
  // interior rows
  for (int n = interior_start; n < interior_end; ++n) {
    std::vector<Index64> Kn(stencil.size());
    for (size_t i = 0; i < stencil.size(); ++i)
      Kn[i] = static_cast<Index64>(n + stencil[i]);
    auto dx = get_dx(Kn, n);
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }
  // boundary 1 rows
  for (int n = b1_start; n < b1_end; ++n) {
    auto Kn = get_indices(n, b1_sten);
    auto dx = get_dx(Kn, n);
    fornberg_row_weights(dx, deriv_order, wrow);
    for (size_t j = 0; j < Kn.size(); ++j)
      trips.emplace_back(static_cast<Index64>(n), Kn[j], wrow[j]);
  }

  Nx_ = Nx;
  D_.resize(Nx_, Nx_);
  D_.setFromTriplets(trips.begin(), trips.end());
  D_.makeCompressed(); // CSR
}

void FiniteDifference1D::apply(const double *y, double *out) const {
  Eigen::Map<const Vec> Y(y, Nx_);
  Eigen::Map<Vec> O(out, Nx_);
  O.noalias() = D_ * Y;
}

void FiniteDifference1D::apply_batch(const double *Y, int ldY, double *Out,
                                     int ldOut, int nvec) const {
  for (int k = 0; k < nvec; ++k) {
    apply(Y + static_cast<size_t>(k) * static_cast<size_t>(ldY),
          Out + static_cast<size_t>(k) * static_cast<size_t>(ldOut));
  }
}

void FiniteDifference1D::triplets(std::vector<Index64> &I,
                                  std::vector<Index64> &J,
                                  std::vector<double> &V) const {
  I.clear();
  J.clear();
  V.clear();
  I.reserve(D_.nonZeros());
  J.reserve(D_.nonZeros());
  V.reserve(D_.nonZeros());
  for (int r = 0; r < D_.rows(); ++r) {
    for (SpMat::InnerIterator it(D_, r); it; ++it) {
      I.push_back(r);
      J.push_back(it.col());
      V.push_back(it.value());
    }
  }
}

} // namespace findiff
} // namespace mathutils