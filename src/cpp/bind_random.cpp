// bind_random.cpp
#include "mathutils/bind/bind_random.hpp"
#include "mathutils/random/random.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// bindings for this class:
//  /**
//    * @brief Default constructor: seeds engine from `std::random_device` via
//    * `std::seed_seq`.
//    * @details Use when non-deterministic seeding is acceptable. Construct
//    once
//    * outside hot loops.
//    */
//   RandomNumberGenerator() : rng_(make_seeded_engine()) {}

//   /**
//    * @brief Deterministic constructor with stream support.
//    * @param seed       User-provided 64-bit seed (reproducible).
//    * @param stream_id  Additional 64-bit stream identifier; different values
//    * produce statistically independent streams under SplitMix64 mixing.
//    */
//   explicit RandomNumberGenerator(uint64_t seed, uint64_t stream_id = 0)
//       : rng_(mixed_seed(seed, stream_id)) {}

// /**
//  * @brief In-place Fisher–Yates shuffle using the internal engine.
//  * @tparam It Random access iterator type.
//  * @param first Begin iterator.
//  * @param last  End iterator.
//  * @note Enables buffer reuse to avoid per-call allocations.
//  */
// template <class It> void shuffle(It first, It last) {
//   std::shuffle(first, last, rng_);
// }

namespace py = pybind11;

void bind_random(py::module_ &m) {
  m.doc() = "Utilities for random number generation.";

  py::class_<mathutils::random::RandomNumberGenerator>(m,
                                                       "RandomNumberGenerator")
      .def(py::init<>())
      .def(py::init<uint64_t, uint64_t>(), py::arg("seed"),
           py::arg("stream_id") = 0,
           R"doc(
           Deterministic constructor with stream support.
            Args:
                seed (int): User-provided 64-bit seed (reproducible).
                stream_id (int): Additional 64-bit stream identifier; different values
                    produce statistically independent streams under SplitMix64 mixing.
           )doc")
      .def("uniform01", &mathutils::random::RandomNumberGenerator::uniform01,
           R"doc(
           Draw a standard uniform in the half-open interval [0,1).
           Returns:
               float: U[0,1).
           )doc")
      .def("uniform_open0_closed1",
           &mathutils::random::RandomNumberGenerator::uniform_open0_closed1,
           R"doc(
           Draw a standard uniform in the interval (0,1].
           Returns:
               float: U(0,1].
           )doc")
      .def("normal01", &mathutils::random::RandomNumberGenerator::normal01,
           R"doc(
           Draw a standard normal N(0,1).
           Returns:
               float: N(0,1).
           )doc")
      .def("exponential",
           &mathutils::random::RandomNumberGenerator::exponential,
           py::arg("rate"),
           R"doc(
           Draw an exponential waiting time with rate λ = rate > 0.
           Args:
               rate (float): Positive rate parameter λ.
           Returns:
               float: Exp(λ) sample.
           )doc")
      .def("kmc_wait", &mathutils::random::RandomNumberGenerator::kmc_wait,
           py::arg("a0"),
           R"doc(
           Convenience alias for KMC waiting time: τ ~ Exp(a0).
           Args:
               a0 (float): Total propensity/rate a0 > 0.
           Returns:
               float: Exponential waiting time.
           )doc")
      .def("random_permutation",
           &mathutils::random::RandomNumberGenerator::random_permutation,
           py::arg("n"),
           R"doc(
           Generate a random permutation of integers [0,n-1].
           Args:
               n (int): Number of elements.
           Returns:
               List[int]: List containing a permutation of 0,...,n-1.
           )doc")
      .def("shuffle",
           &mathutils::random::RandomNumberGenerator::shuffle<
               std::vector<std::size_t>::iterator>,
           py::arg("first"), py::arg("last"),
           R"doc(
           In-place Fisher–Yates shuffle using the internal engine.
           Args:
               first (iterator): Begin iterator.
               last (iterator): End iterator.
           )doc")
      .def("reseed", &mathutils::random::RandomNumberGenerator::reseed,
           py::arg("seed"), py::arg("stream_id") = 0,
           R"doc(
           Reseed the engine deterministically.
           Args:
               seed (int): New 64-bit seed.
               stream_id (int): New 64-bit stream identifier.
           )doc");
}