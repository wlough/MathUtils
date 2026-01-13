#pragma once

/**
 * @file random.hpp
 * @brief Utilities for random number generation (64-bit Mersenne Twister
 * engine).
 */

#include <algorithm> // shuffle
#include <cassert>   // assert
#include <cmath>     // log, nextafter
#include <cstdint>   // uint64_t
#include <numeric>   // iota
#include <random>    // mt19937_64, random_device, distributions
#include <vector>    // vector

namespace mathutils {
namespace random {

/**
 * @namespace mathutils::random::detail
 * @brief Internal helpers (not part of the public API).
 */
namespace detail {
/**
 * @brief SplitMix64 mixer for seeding/stream splitting.
 * @param x 64-bit input.
 * @return Mixed 64-bit value.
 */
constexpr inline uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
} // namespace detail

/**
 * @class RandomNumberGenerator
 * @brief Minimal RNG wrapper for simulations (hot-loop friendly).
 *
 * Typical use:
 * @code
 * // seed=1234, stream_id=0 (default)
 * RandomNumberGenerator rng(1234, 0);
 * double u = rng.uniform01();                 // U[0,1)
 * double z = rng.normal01();                  // N(0,1)
 * double tau = rng.kmc_wait(a0);              // exponential wait with rate a0
 * std::vector<size_t> p = rng.random_permutation(N); // shuffled 0..N-1
 * @endcode
 */
class RandomNumberGenerator {
public:
  /// Underlying PRNG engine type.
  using engine_type = std::mt19937_64;

  /**
   * @brief Default constructor: seeds engine from `std::random_device` via
   * `std::seed_seq`.
   * @details Use when non-deterministic seeding is acceptable. Construct once
   * outside hot loops.
   */
  RandomNumberGenerator() : rng_(make_seeded_engine()) {}

  /**
   * @brief Deterministic constructor with stream support.
   * @param seed       User-provided 64-bit seed (reproducible).
   * @param stream_id  Additional 64-bit stream identifier; different values
   * produce statistically independent streams under SplitMix64 mixing.
   */
  explicit RandomNumberGenerator(uint64_t seed, uint64_t stream_id = 0)
      : rng_(mixed_seed(seed, stream_id)) {}

  /**
   * @brief Draw a standard uniform in the half-open interval [0,1).
   * @return U[0,1).
   */
  double uniform01() const { return U01_(rng_); }

  /**
   * @brief Draw a standard uniform in the interval (0,1].
   * @return U(0,1].
   * @details Useful for \f$-\log(u)\f$ to avoid \f$\log(0)\f$. Implemented by
   *          replacing 0 with `nextafter(0,1)`.
   */
  double uniform_open0_closed1() const {
    double u = U01_(rng_);
    if (u == 0.0)
      u = std::nextafter(0.0, 1.0);
    return u;
  }

  /**
   * @brief Draw a standard normal N(0,1).
   * @return N(0,1).
   */
  double normal01() const { return N01_(rng_); }

  /**
   * @brief Draw an exponential waiting time with rate \f$\lambda =\text{rate} >
   * 0\f$.
   * @param rate Positive rate parameter \f$\lambda\f$.
   * @return \f$\mathrm{Exp}(\lambda)\f$ sample.
   * @pre `rate > 0.0` (checked by `assert`).
   * @note Uses inverse-CDF: \f$-\log(u)/\lambda\f$ with \f$u\in(0,1]\f$.
   */
  double exponential(double rate) const {
    assert(rate > 0.0);
    const double u = uniform_open0_closed1();
    return -std::log(u) / rate;
  }

  /**
   * @brief Convenience alias for KMC waiting time: \f$\tau \sim
   * \mathrm{Exp}(a_0)\f$.
   * @param a0 Total propensity/rate \f$a_0>0\f$.
   * @return Exponential waiting time.
   * @pre `a0 > 0.0` (checked in `exponential`).
   */
  double kmc_wait(double a0) const { return exponential(a0); }

  /**
   * @brief Generate a random permutation of integers \f$[0,n-1]\f$.
   * @param n Number of elements.
   * @return Vector containing a permutation of \f$0,\dots,n-1\f$.
   * @complexity \f$O(n)\f$.
   * @note For high-frequency use, prefer keeping a buffer and
   * `shuffle(first,last)`.
   */
  std::vector<std::size_t> random_permutation(std::size_t n) {
    std::vector<std::size_t> p(n);
    std::iota(p.begin(), p.end(), std::size_t{0});
    std::shuffle(p.begin(), p.end(), rng_);
    return p;
  }

  /**
   * @brief In-place Fisher–Yates shuffle using the internal engine.
   * @tparam It Random access iterator type.
   * @param first Begin iterator.
   * @param last  End iterator.
   * @note Enables buffer reuse to avoid per-call allocations.
   */
  template <class It> void shuffle(It first, It last) {
    std::shuffle(first, last, rng_);
  }

  /**
   * @brief Reseed the engine deterministically.
   * @param seed      New 64-bit seed.
   * @param stream_id New 64-bit stream identifier.
   * @warning Reseeding resets the sequence; use only when starting a new
   * stream.
   */
  void reseed(uint64_t seed, uint64_t stream_id = 0) {
    rng_.seed(mixed_seed(seed, stream_id));
  }

private:
  /**
   * @brief Build an engine seeded from `std::random_device` via
   * `std::seed_seq`.
   * @return Seeded engine.
   * @note Called only by the default constructor.
   */
  static engine_type make_seeded_engine() {
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    return engine_type(seq);
  }

  /**
   * @brief Combine `(seed, stream_id)` into an engine seed using SplitMix64.
   * @param seed 64-bit user seed.
   * @param stream_id 64-bit stream identifier.
   * @return Engine seed value.
   */
  static engine_type::result_type mixed_seed(uint64_t seed,
                                             uint64_t stream_id) {
    const uint64_t mixed =
        detail::splitmix64(seed) ^
        detail::splitmix64(stream_id + 0x9e3779b97f4a7c15ULL);
    return static_cast<engine_type::result_type>(mixed);
  }

  // Engine and distributions are mutable so const draw methods can advance
  // state.
  mutable engine_type rng_; ///< PRNG engine.
  mutable std::uniform_real_distribution<double> U01_{
      0.0, 1.0}; ///< Cached U[0,1) distribution.
  mutable std::normal_distribution<double> N01_{
      0.0, 1.0}; ///< Cached N(0,1) distribution.
};

} // namespace random
} // namespace mathutils
