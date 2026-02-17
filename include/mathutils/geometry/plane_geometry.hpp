#pragma once

/**
 * @file geometry.hpp
 * @brief Geometric utilities
 */

#include <cmath>   // For M_PI and std::acos
#include <utility> // std::swap

namespace mathutils {
namespace geometry {

/**
 * @brief Compute triangle area from edge lengths using numerically stable
 * variant of Heron's formula.
 * @param a
 * @param b
 * @param c
 * @return double
 *
 * See https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
 */
double inline heron_area(const double &L1, const double &L2, const double &L3) {
  double a = L1, b = L2, c = L3;
  if (b > a) {
    std::swap(a, b);
  }
  if (c > a) {
    std::swap(a, c);
  }
  if (c > b) {
    std::swap(b, c);
  }
  return std::sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) *
                   (a + (b - c))) /
         4;
}
//
} // namespace geometry
} // namespace mathutils
