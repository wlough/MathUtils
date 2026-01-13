#pragma once

/**
 * @file findiff.hpp
 * @brief Finite difference utilities
 */

#include <Eigen/Core> // Eigen::MatrixXd, Eigen::VectorXd
#include <Eigen/Sparse>
#include <optional>
#include <stdexcept> // std::invalid_argument
#include <vector>    // std::vector

/////////////////////////////////////
/////////////////////////////////////
// Finite difference utilities //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace findiff {

/**
 * @brief Compute Fornberg weights for finite difference derivatives.
 *
 * Calculates weights for derivatives 0 to d_max at x0 on nodes x_stencil
 * using Fornberg's algorithm. The returned matrix W has shape (d_max+1, n),
 * where W(d,j) is the weight for f(x_j) when computing the d-th derivative.
 *
 * The d-th derivative at x0 is computed as:
 * (D^d f)(x0) = Sum_j( W(d,j) * f(x_stencil[j]) )
 *
 * @param x_stencil Grid points to use for the stencil
 * @param x0 Point where derivative is evaluated
 * @param d_max Maximum derivative order to calculate weights for
 * @return Matrix of weights with shape (d_max+1, n)
 * @throws std::invalid_argument if d_max < 0 or x_stencil is empty
 */
Eigen::MatrixXd fornberg_weights(const Eigen::VectorXd &x_stencil, double x0,
                                 int d_max);

/**
 * @brief Compute Fornberg weights for finite difference derivatives
 * (std::vector version).
 *
 * @param x_stencil Grid points to use for the stencil
 * @param x0 Point where derivative is evaluated
 * @param d_max Maximum derivative order to calculate weights for
 * @return Matrix of weights with shape (d_max+1, n)
 * @throws std::invalid_argument if d_max < 0 or x_stencil is empty
 */
Eigen::MatrixXd fornberg_weights(const std::vector<double> &x_stencil,
                                 double x0, int d_max);

} // namespace findiff
} // namespace mathutils

/////////////////////////////////////
/////////////////////////////////////
// FinDiff Operator //////
/////////////////////////////////////
/////////////////////////////////////
namespace mathutils {
namespace findiff {

using Index64 = long long; // matches NumPy default index width
using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, Index64>;
using Vec = Eigen::VectorXd;

struct BuildSpec {
  std::vector<double> x;        // grid points (size Nx)
  std::vector<int> stencil;     // offsets, e.g., {-2,-1,0,1,2}
  int deriv_order = 1;          // >=0
  std::optional<double> period; // None => nonperiodic
  std::optional<std::vector<int>> boundary0_stencil;
  std::optional<std::vector<int>> boundary1_stencil;
};

struct BuildSpecPeriodic {
  std::vector<double> x;    // grid points (size Nx)
  std::vector<int> stencil; // offsets, e.g., {-2,-1,0,1,2}
  int deriv_order = 1;      // >=0
  double period;            // period length
};

struct BuildSpecNonPeriodic {
  std::vector<double> x;              // grid points (size Nx)
  std::vector<int> interior_stencil;  // offsets, e.g., {-2,-1,0,1,2}
  std::vector<int> boundary0_stencil; // offsets, e.g., {0,1,2,3,4}
  std::vector<int> boundary1_stencil; // offsets, e.g., {-4,-3,-2,-1,0}
  int deriv_order = 1;                // >=0
};

/**
 * 1D finite-difference operator using Fornberg weights.
 * Build once, apply many times. Stores a CSR (RowMajor) sparse matrix.
 */
class FiniteDifference1D {
public:
  FiniteDifference1D() = default;
  explicit FiniteDifference1D(const BuildSpec &spec) { build(spec); }

  void build(const BuildSpec &spec);
  void build(const BuildSpecPeriodic &spec);
  void build(const BuildSpecNonPeriodic &spec);

  int size() const noexcept { return Nx_; }

  // y (length Nx) -> out (length Nx)
  void apply(const double *y, double *out) const;

  // Batched apply: each column is a vector, leading dim = Nx
  void apply_batch(const double *Y, int ldY, double *Out, int ldOut,
                   int nvec) const;

  // Get CSR triplets for Python/SciPy construction if desired
  void triplets(std::vector<Index64> &I, std::vector<Index64> &J,
                std::vector<double> &V) const;

  // Direct access for C++
  const SpMat &matrix() const { return D_; }

private:
  int Nx_ = 0;
  SpMat D_;
};

} // namespace findiff
} // namespace mathutils