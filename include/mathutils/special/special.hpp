#pragma once

/**
 * @file special.hpp
 * @brief Special functions
 */

#include "mathutils/special/log_factorial_lookup_table.hpp"
#include <cmath>

/////////////////////////////////////
/////////////////////////////////////
// Special functions ////////////////
/////////////////////////////////////
/////////////////////////////////////

// Some inline template functions
namespace mathutils {
namespace special {
/**
 * @brief Compute \f$(-1)^n\f$ for integer n.
 */
inline int minus_one_to_int_pow(int n) {
  // Returns -1 if n is odd, 1 if n is even
  return 1 - 2 * (n & 1);
}

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

} // namespace special
} // namespace mathutils
