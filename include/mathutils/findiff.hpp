#pragma once

/**
 * @file findiff.hpp
 * @brief Finite difference utilities
 */

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

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
