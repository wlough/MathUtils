
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
/////////////////////////////////////
// namespace mathutils {

// namespace special {

// inline double phi_independent_Yklm(int k, int l, int m, double theta) {

//   // int abs_m = std::abs(m);
//   // if ((theta == M_PI_2) && (l - abs_m - 2 * k != 0)) {
//   //   return 0.0;
//   // }
//   // if (((theta == 0.0) || (theta == M_PI)) &&
//   //     (abs_m + 2 * k != 0)) { // only zero when m=k=0
//   //   return 0.0;
//   // }
//   // int s = 1;
//   // if ((m > 0) && (m % 2 == 1)) {
//   //   s *= -1;
//   // }
//   // if (k % 2 == 1) {
//   //   s *= -1;
//   // }
//   // if ((theta > M_PI_2) && (l - abs_m - 2 * k % 2 == 1)) {
//   //   s *= -1;
//   // }
//   // double log_stuff =
//   //     -0.5 * std::log(4 * M_PI) + 0.5 * std::log(2 * l + 1) +
//   //     0.5 * log_factorial(l + abs_m) + 0.5 * log_factorial(l - abs_m) -
//   //     (abs_m + 2 * k) * std::log(2) - log_factorial(l - abs_m - 2 * k) -
//   //     log_factorial(abs_m + k) - log_factorial(k);
//   // if (theta != M_PI_2) {
//   //   double cos_theta = std::cos(theta);
//   //   double abs_cos_theta = std::abs(cos_theta);
//   //   log_stuff += (l - abs_m - 2 * k) * std::log(abs_cos_theta);
//   // }
//   // if ((theta != 0.0) && (theta != M_PI)) {
//   //   double sin_theta = std::sin(theta);
//   //   double abs_sin_theta = std::abs(sin_theta);
//   //   log_stuff += (abs_m + 2 * k) * std::log(abs_sin_theta);
//   // }
//   // return s * std::exp(log_stuff);
//   int abs_m = std::abs(m);
//   double cos_theta = std::cos(theta);
//   double sin_theta = std::sin(theta);
//   double ReLogRe_Yklm =
//       -0.5 * std::log(4 * M_PI) + 0.5 * std::log(2 * l + 1) +
//       0.5 * log_factorial(l + abs_m) + 0.5 * log_factorial(l - abs_m) -
//       (abs_m + 2 * k) * std::log(2) - log_factorial(l - abs_m - 2 * k) -
//       log_factorial(abs_m + k) - log_factorial(k) +
//       (l - abs_m - 2 * k) * ReLog(cos_theta) +
//       (abs_m + 2 * k) * ReLog(sin_theta);
//   int ImLogRe_over_pi_Yklm =
//       k + (m + abs_m) / 2 + (l - abs_m) * (cos_theta < 0) & 1;

//   return (ImLogRe_over_pi_Yklm & 1) ? -std::exp(ReLogRe_Yklm)
//                                     : std::exp(ReLogRe_Yklm);
// }

// /**
//  * @brief Compute \f$Re(Log(\prod_k x_k^{p_k}))\f$ and \f$Im(Log(\prod_k
//  * x_k^{p_k}))/\pi\f$ for \f$x_k\in \mathbb{R}\f$ and \f{p_k\in
//  \mathbb{Z}\f$.
//  *
//  * \f[
//  * \mathbb{R}^N\times\mathbb{Z}^N\rightarrow
//  * \mathbb{R}\cup\lbrace-\infty\rbrace\times\mathbb{Z}_{2}\\
//  * \f]
//  *
//  * @param bases BaseContainer of real base values
//  * @param exponents ExponentContainer of integer exponent values
//  * @return std::pair<double, int> {re_log, arg_over_pi}
//  *
//  * @note bases[i] == 0 and exponents[i] > 0, contributes −∞ to re_log
//  * @note bases[i] == 0 and exponents[i] < 0, contributes +∞ to re_log
//  * @note For bases[i] < 0, arg(bases[i]) = π so each negative base
//  contributes
//  *      exponents[i]*\f$\pi\f$ to the total argument.
//  */
// template <typename BaseContainer, typename ExponentContainer>
// inline std::pair<double, int>
// ReLog_ImLog_over_pi_of_product_real_bases_and_integer_powers(
//     const BaseContainer &bases, const ExponentContainer &exponents) {

//   double re_log = 0.0;
//   int arg_accumulator = 0; // initialize to int zero
//   auto base_it = bases.begin();
//   auto exp_it = exponents.begin();
//   while (base_it != bases.end() && exp_it != exponents.end()) {
//     double b = static_cast<double>(*base_it);
//     int p = static_cast<int>(*exp_it);
//     // Skip zero exponents (b^0 = 1)
//     if (p == 0) {
//       ++base_it;
//       ++exp_it;
//       continue;
//     }
//     re_log += static_cast<double>(p) * ReLog(b);
//     if (b < 0.0) {
//       arg_accumulator += p;
//     }
//     ++base_it;
//     ++exp_it;
//   }
//   return {re_log, arg_accumulator & 1};
// }

// /**
//  * @brief Compute real/imaginary parts of logarithm of ∏ᵢ
//  bases[i]**exponents[i]
//  * for real bases and real/integer exponents.
//  *
//  * @param bases BaseContainer of real base values
//  * @param exponents ExponentContainer of real/integer exponent values
//  * @return std::pair<double, double/int> {re_log, arg_over_pi}
//  *         - re_log: ∑ᵢ exponents[i] * log|bases[i]|, the real part of the
//  log
//  *         - arg_over_pi: (∑ᵢ exponents[i] * arg(bases[i])) / π, reduced mod
//  2
//  *
//  * @note If bases[i] == 0 and exponents[i] > 0, contributes −∞ to re_log
//  * @note If bases[i] == 0 and exponents[i] < 0, contributes +∞ to re_log
//  * @note For bases[i] < 0, arg(bases[i]) = π so each negative base
//  contributes
//  *       to the argument according to the exponent
//  */
// template <typename BaseContainer, typename ExponentContainer>
// inline auto log_of_product_of_powers(const BaseContainer &bases,
//                                      const ExponentContainer &exponents) {

//   using BaseType = typename BaseContainer::value_type;
//   using ExponentType = typename ExponentContainer::value_type;

//   // Size check for containers that support it
//   // if constexpr (requires {
//   //                 bases.size();
//   //                 exponents.size();
//   //               }) {
//   //   if (bases.size() != exponents.size()) {
//   //     throw std::invalid_argument(
//   //         "bases and exponents must have the same size");
//   //   }
//   // }

//   double re_log = 0.0;
//   ExponentType
//       arg_accumulator{}; // initialize to ExponentType zero (either 0 or 0.0)

//   auto base_it = bases.begin();
//   auto exp_it = exponents.begin();

//   while (base_it != bases.end() && exp_it != exponents.end()) {
//     double b = static_cast<double>(*base_it);
//     auto p = *exp_it;

//     if (p == ExponentType{}) {
//       // Skip zero exponents (b^0 = 1)
//       ++base_it;
//       ++exp_it;
//       continue;
//     }
//     // re_log += static_cast<double>(p) * ReLog(b);
//     if (b == 0.0) {
//       re_log +=
//           -static_cast<double>(p) * std::numeric_limits<double>::infinity();
//     } else if (b < 0.0) {
//       re_log += static_cast<double>(p) * std::log(-b);
//       arg_accumulator += p;
//     } else {
//       re_log += static_cast<double>(p) * std::log(b);
//     }

//     ++base_it;
//     ++exp_it;
//   }

//   if constexpr (std::is_integral_v<ExponentType>) {
//     // Integer exponents: use fast parity check
//     double arg_over_pi = static_cast<double>(arg_accumulator & 1);
//     return std::make_pair(re_log, arg_over_pi);
//   } else {
//     // Floating point exponents: use modulo
//     double arg_over_pi =
//     std::fmod(static_cast<double>(arg_accumulator), 2.0); return
//     std::make_pair(re_log, arg_over_pi);
//   }
// }

// // template <typename Container1, typename Container2>
// // inline std::pair<double, double>
// // log_of_product_of_powers_template(const Container1 &bases,
// //                                   const Container2 &exponents) {

// //   double re_log = 0.0;
// //   double arg_over_pi = 0.0;

// //   auto it_bases = bases.begin();
// //   auto it_exponents = exponents.begin();

// //   while (it_bases != bases.end() && it_exponents != exponents.end()) {
// //     double b = *it_bases;
// //     double p = *it_exponents;

// //     if (b == 0.0) {
// //       if (p != 0.0) {
// //         re_log += -p * std::numeric_limits<double>::infinity();
// //       }
// //     } else if (b < 0.0) {
// //       re_log += p * std::log(-b);
// //       arg_over_pi += p;
// //     } else {
// //       re_log += p * std::log(b);
// //     }

// //     ++it_bases;
// //     ++it_exponents;
// //   }

// //   arg_over_pi = std::fmod(arg_over_pi, 2.0);

// //   return {re_log, arg_over_pi};
// // }
// } // namespace special
// } // namespace mathutils

// namespace mathutils {

// /**
//  * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for small sin(theta) and
//  * cos(theta).
//  * @param l Degree of the spherical harmonic.
//  * @param m Order of the spherical harmonic.
//  * @param theta Polar angle in radians.
//  * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
//  */
// double phi_independent_problem_angle_small_Ylm(int l, int m, double theta) {
//   double cos_theta = std::cos(theta);
//   double sin_theta = std::sin(theta);
//   int abs_m = std::abs(m);
//   double rlm = 1.0 / (2.0 * std::sqrt(M_PI));
//   if (m > 0) {
//     if (m % 2 == 1) {
//       rlm *= -1;
//     }
//   }

//   double log_qk = 0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m)
//   -
//                   abs_m * std::log(2) - 0.5 * log_factorial(l - abs_m) -
//                   log_factorial(abs_m);
//   double trig_term =
//       std::pow(cos_theta, l - abs_m) * std::pow(sin_theta, abs_m);
//   double sk = 1;
//   double sum_Qk = sk * std::exp(log_qk) * trig_term;
//   double minus_log4 = std::log(0.25);

//   for (int k = 1; k <= (l - abs_m) / 2; ++k) {
//     sk *= -1;
//     log_qk += minus_log4 + std::log(l - abs_m - 2 * k + 2) +
//               std::log(l - abs_m - 2 * k + 1) - std::log(abs_m + k) -
//               std::log(k);
//     trig_term = std::pow(cos_theta, l - abs_m - 2 * k) *
//                 std::pow(sin_theta, abs_m + 2 * k);

//     sum_Qk += sk * std::exp(log_qk) * trig_term;
//   }

//   return rlm * sum_Qk;
// }

// /**
//  * @brief Compute Ylm(theta, phi)/exp(i*theta*phi) for sin(theta) ~
//  * +-cos(theta).
//  * @param l Degree of the spherical harmonic.
//  * @param m Order of the spherical harmonic.
//  * @param theta Polar angle in radians.
//  * @param phi Azimuthal angle in radians.
//  * @return Value of Ylm(theta, phi)/exp(i*theta*phi)
//  */
// double phi_independent_problem_angle_same_Ylm(int l, int m, double theta,
//                                               double phi) {
//   double cos_theta = std::cos(theta);
//   double tan_theta = std::tan(theta);
//   double abs_cos_theta = std::abs(cos_theta);
//   double abs_tan_theta = std::abs(tan_theta);
//   double log_abs_cos_theta = std::log(abs_cos_theta);
//   double log_abs_tan_theta = std::log(abs_tan_theta);
//   int abs_m = std::abs(m);
//   double trig_pows = std::pow(cos_theta, l) * std::pow(tan_theta, abs_m);
//   double rlm = sign(trig_pows) / (2 * std::sqrt(M_PI));
//   if (m > 0) {
//     if (m % 2 == 1) {
//       rlm *= -1;
//     }
//   }

//   double log_qk = 0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m)
//   +
//                   l * log_abs_cos_theta + abs_m * log_abs_tan_theta -
//                   abs_m * std::log(2) - 0.5 * log_factorial(l - abs_m) -
//                   log_factorial(abs_m);
//   double sk = 1;
//   double sum_Qk = sk * std::exp(log_qk);
//   double minus_log4 = std::log(0.25);
//   for (int k = 1; k <= (l - abs_m) / 2; ++k) {
//     sk *= -1;
//     log_qk +=
//         2 * log_abs_tan_theta + minus_log4 + std::log(l - abs_m - 2 * k + 2)
//         + std::log(l - abs_m - 2 * k + 1) - std::log(abs_m + k) -
//         std::log(k);
//     sum_Qk += sk * std::exp(log_qk);
//   }
//   return rlm * sum_Qk;
// }
// // computes a0**n0*a1**n1*...*ak**nk as exp(log(a0**n0*a1**n1*...*ak**nk))=
// // double fun(double a, int[] n, int k) {
// //   double log_product = 0.0;
// //   for (int i = 0; i < k; ++i) {
// //     log_product += n[i] * std::log(a[i]);
// //   }
// //   return std::exp(log_product);
// // }

// inline double magnitude_Ylm(int l, int abs_m, double abs_cos_theta) {
//   double epsilon = 1e-8;
//   double minus_log4 = std::log(0.25);

//   int ell_minus_m = l - abs_m;
//   double abs_sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);

//   double rlm = 1.0 / (2.0 * std::sqrt(M_PI));
//   int k_bound = (ell_minus_m) / 2;
//   int sk = 1;
//   double log_qk = 0.5 * std::log(2 * l + 1) + 0.5 * log_factorial(l + abs_m)
//   -
//                   abs_m * std::log(2) - 0.5 * log_factorial(ell_minus_m) -
//                   log_factorial(abs_m);

//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   // If cos(theta) ~ +/-sin(theta) replace sin(theta) with tan(theta)
//   if (std::abs(abs_cos_theta - abs_sin_theta) < epsilon) {
//     double abs_tan_theta = abs_sin_theta / abs_cos_theta;
//     double log_abs_cos_theta = std::log(abs_cos_theta);
//     double log_abs_tan_theta = std::log(abs_tan_theta);
//     log_qk += l * log_abs_cos_theta + abs_m * log_abs_tan_theta;
//     double sum_Qk = sk * std::exp(log_qk);
//     for (int k = 1; k <= k_bound; ++k) {
//       sk *= -1;
//       log_qk += 2 * log_abs_tan_theta;
//       log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
//                 std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
//                 std::log(k);
//       sum_Qk += sk * std::exp(log_qk);
//     }
//     return rlm * sum_Qk;
//   }

//   // if cos(theta) ~ 0, sin(theta) ~ 1
//   if (abs_cos_theta < epsilon) {
//     double log_abs_sin_theta = std::log(abs_sin_theta);
//     log_qk += abs_m * log_abs_sin_theta;
//     double abs_cos_theta_pow = std::pow(abs_cos_theta, ell_minus_m);
//     double sum_Qk = sk * std::exp(log_qk) * abs_cos_theta_pow;
//     for (int k = 1; k <= k_bound; ++k) {
//       sk *= -1;
//       log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
//                 std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
//                 std::log(k);
//       abs_cos_theta_pow = std::pow(abs_cos_theta, ell_minus_m - 2 * k);
//       sum_Qk += sk * std::exp(log_qk) * abs_cos_theta_pow;
//     }

//     return rlm * sum_Qk;
//   }

//   // if sin(theta) ~ 0, cos(theta) ~ +/-1
//   if (abs_sin_theta < epsilon) {
//     double trig_term =
//         std::pow(abs_cos_theta, ell_minus_m) * std::pow(abs_sin_theta,
//         abs_m);
//     double sum_Qk = sk * std::exp(log_qk) * trig_term;
//     for (int k = 1; k <= k_bound; ++k) {
//       sk *= -1;
//       log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
//                 std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
//                 std::log(k);
//       trig_term = std::pow(abs_cos_theta, ell_minus_m - 2 * k) *
//                   std::pow(abs_sin_theta, abs_m + 2 * k);
//       sum_Qk += sk * std::exp(log_qk) * trig_term;
//     }
//     return rlm * sum_Qk;
//   }

//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   ////////////////////////////////////////////////////////
//   // if log(cos(theta)) and log(sin(theta)) are defined
//   double log_abs_cos_theta = std::log(abs_cos_theta);
//   double log_abs_sin_theta = std::log(abs_sin_theta);
//   log_qk += ell_minus_m * log_abs_cos_theta + abs_m * log_abs_sin_theta;
//   double sum_Qk = sk * std::exp(log_qk);
//   for (int k = 1; k <= k_bound; ++k) {
//     sk *= -1;
//     log_qk += -2 * log_abs_cos_theta + 2 * log_abs_sin_theta;
//     log_qk += minus_log4 + std::log(ell_minus_m - 2 * k + 2) +
//               std::log(ell_minus_m - 2 * k + 1) - std::log(abs_m + k) -
//               std::log(k);
//     sum_Qk += sk * std::exp(log_qk);
//   }

//   return rlm * sum_Qk;
// }

// inline std::complex<double> Ylm(int l, int m, double theta, double phi) {
//   if (theta > M_PI || theta < 0.0) {
//     throw std::out_of_range("theta must be in the range [0, pi]");
//   }
//   return std::exp(std::complex<double>(0, m * phi)) *
//          phi_independent_Ylm(l, m, theta);
//   // if (theta > M_PI || theta < 0.0) {
//   //   throw std::out_of_range("theta must be in the range [0, pi]");
//   // }
//   // int abs_m = std::abs(m);
//   // double cos_theta = std::cos(theta);
//   // double abs_cos_theta = std::abs(cos_theta);
//   // std::complex<double> val = std::exp(std::complex<double>(0, m * phi)) *
//   //                            magnitude_Ylm(l, abs_m, abs_cos_theta);
//   // if ((m > 0) && (abs_m % 2 == 1)) {
//   //   val *= -1;
//   // }
//   // if ((cos_theta < 0) && (l - abs_m % 2 == 1)) {
//   //   val *= -1;
//   // }

//   // return val;
// }

// Eigen::VectorXcd Ylm_vectorized(int l, int m, const Eigen::VectorXd &theta,
//                                 const Eigen::VectorXd &phi) {
//   if (l < 0) {
//     throw std::out_of_range("l must be non-negative");
//   }
//   if (m < -l || m > l) {
//     throw std::out_of_range("m must be in the range [-l, l]");
//   }
//   if (theta.size() != phi.size()) {
//     throw std::invalid_argument("theta and phi must have the same size");
//   }
//   Eigen::VectorXcd result(theta.size());
//   for (int i = 0; i < theta.size(); ++i) {
//     result[i] = Ylm(l, m, theta[i], phi[i]);
//   }
//   return result;
// }

// // inline double real_Ylm(int l, int m, double theta, double phi) {

// //   if (theta > M_PI || theta < 0.0) {
// //     throw std::out_of_range("theta must be in the range [0, pi]");
// //   }
// //   int abs_m = std::abs(m);
// //   double cos_theta = std::cos(theta);
// //   double abs_cos_theta = std::abs(cos_theta);
// //   double val = magnitude_Ylm(l, abs_m, abs_cos_theta);
// //   int s = 1;

// //   if (m < 0) {
// //     val *= std::sqrt(2) * std::sin(abs_m * phi);
// //     if ((cos_theta < 0) && (l - abs_m % 2 == 1)) {
// //       s *= -1;
// //     }
// //     if (abs_m % 2 == 1) {
// //       s *= -1;
// //     }
// //     return s * val;
// //   }
// //   if (m == 0) {
// //     if ((cos_theta < 0) && (l % 2 == 1)) {
// //       s *= -1;
// //     }
// //     return s * val;
// //   }
// //   val *= std::sqrt(2) * std::cos(abs_m * phi);
// //   if ((cos_theta < 0) && (l - abs_m % 2 == 1)) {
// //     s *= -1;
// //   }
// //   return s * val;
// // }

// Eigen::VectorXd real_Ylm_vectorized(int l, int m, const Eigen::VectorXd
// &theta,
//                                     const Eigen::VectorXd &phi) {
//   if (l < 0) {
//     throw std::out_of_range("l must be non-negative");
//   }
//   if (m < -l || m > l) {
//     throw std::out_of_range("m must be in the range [-l, l]");
//   }
//   if (theta.size() != phi.size()) {
//     throw std::invalid_argument("theta and phi must have the same size");
//   }
//   Eigen::VectorXd result(theta.size());
//   for (int i = 0; i < theta.size(); ++i) {
//     result[i] = real_Ylm(l, m, theta[i], phi[i]);
//   }
//   return result;
// }

// inline int spherical_harmonic_s_lmtheta(int l, int m, double theta) {
//   int sign_m = (m > 0) - (m < 0);
//   int sign_cos_theta = (theta < M_PI_2) - (theta > M_PI_2);
//   int s = 1;
//   if (l % 2 == 1) {
//     s *= sign_cos_theta;
//   }
//   if (m % 2 == 1) {
//     s *= -sign_m * sign_cos_theta;
//   }
//   return s;
// }

// inline int sign_phi_independent_Yklm(int k, int l, int m, double theta) {

//   int s = 1;
//   if ((m > 0) && (m % 2 == 1)) {
//     s *= -1;
//   }
//   if (k % 2 == 1) {
//     s *= -1;
//   }
//   if ((theta > M_PI_2) && (l - std::abs(m) - 2 * k % 2 == 1)) {
//     s *= -1;
//   }
//   if ((theta == M_PI_2) && (l - std::abs(m) - 2 * k != 0)) {
//     s *= 0;
//   }
//   if (((theta == 0.0) || (theta == M_PI)) && (std::abs(m) + 2 * k != 0)) {
//     s *= 0;
//   }

//   return s;
// }

// // inline double log_abs_phi_independent_Yklm(int k, int l, int m, double
// theta)
// // {
// //   int s = sign_phi_independent_Yklm(k, l, m, theta);
// // }

// inline double log_abs_Nklm(int k, int l, int abs_m) {
//   return -0.5 * std::log(4 * M_PI) + 0.5 * std::log(2 * l + 1) +
//          0.5 * log_factorial(l + abs_m) + 0.5 * log_factorial(l - abs_m) -
//          (abs_m + 2 * k) * std::log(2) - log_factorial(l - abs_m - 2 * k) -
//          log_factorial(abs_m + k) - log_factorial(k);
// }

// } // namespace mathutils