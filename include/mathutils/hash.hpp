#pragma once

/**
 * @file hash.hpp
 * @brief Hashing utilities for use with std::unordered_map` /
 * `std::unordered_set`.
 *
 * Provides:
 *   - `hash_combine(seed, value)`: a non-cryptographic combiner for incremental
 * hashing
 *   - `hash_combine(values)`: combine an array of `size_t` hash values into one
 * hash
 *   - `ArrayHash<T, N>`: a hasher for `std::array<T, N>` (order-sensitive)
 *
 */

#include <array>
#include <bit>        // std::rotl
#include <cstddef>    // std::size_t
#include <functional> // std::hash
#include <type_traits>

namespace mathutils {
namespace hash {

/**
 * @brief Mix a new hash value into an existing seed (order-sensitive).
 *
 * This function updates `seed` by combining it with `value` using a lightweight
 * mixing scheme designed to reduce collisions for structured inputs (e.g.,
 * small integers, sequential IDs).
 *
 * Typical usage:
 * @code
 * std::size_t seed = 0;
 * hash_combine(seed, std::hash<int>{}(x));
 * hash_combine(seed, std::hash<int>{}(y));
 * @endcode
 *
 * @param[in,out] seed  Accumulated hash state to be updated.
 * @param[in] value     Hash value to mix into the seed.
 */
inline void hash_combine(std::size_t &seed, std::size_t value) noexcept {
  if constexpr (sizeof(std::size_t) == 8) {
    // Odd constants; multiplication promotes diffusion on 64-bit.
    constexpr std::size_t k1 = 0x9e3779b97f4a7c15ULL;
    constexpr std::size_t k2 = 0xbf58476d1ce4e5b9ULL;

    seed ^= value + k1;
    seed = std::rotl(seed, 27) * k2;
    seed ^= seed >> 33;
  } else {
    // 32-bit constants (Murmur3-style).
    constexpr std::size_t k1 = 0x9e3779b9UL;
    constexpr std::size_t k2 = 0x85ebca6bUL;

    seed ^= value + k1;
    seed = std::rotl(seed, 13) * k2;
    seed ^= seed >> 16;
  }
}

/**
 * @brief Combine an array of hash values into a single hash (order-sensitive).
 *
 * This is a convenience wrapper for combining multiple pre-hashed values.
 *
 * @tparam N  Number of values.
 * @param[in] values  Array of hash values to combine in order.
 * @return Combined hash.
 */
template <std::size_t N>
inline std::size_t
hash_combine(const std::array<std::size_t, N> &values) noexcept {
  std::size_t seed = 0;
  for (std::size_t v : values) {
    hash_combine(seed, v);
  }
  return seed;
}

/**
 * @brief Hasher for `std::array<T, N>` (order-sensitive).
 *
 * Hashes each element using `std::hash<T>` and combines them in index order.
 * If you want order-insensitive hashing (treat the array as a set),
 * canonicalize (e.g., sort) the array before hashing, or implement a separate
 * hasher.
 *
 * @tparam T  Element type (must be hashable by `std::hash<T>`).
 * @tparam N  Array size.
 */
template <typename T, std::size_t N> struct ArrayHash {
  std::size_t operator()(std::array<T, N> const &a) const noexcept {
    static_assert(std::is_invocable_r_v<std::size_t, std::hash<T>, const T &>,
                  "ArrayHash<T,N>: std::hash<T> must be invocable on `const "
                  "T&` and return size_t.");

    std::size_t seed = 0;
    std::hash<T> h{};
    for (const T &x : a) {
      mathutils::hash::hash_combine(seed, h(x));
    }
    return seed;
  }
};

} // namespace hash
} // namespace mathutils
