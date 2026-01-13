from numba import njit
import numpy as np
import math

factorial_table = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype="int64",
)

log_factorial_table = np.array(
    [
        0.0,
        0.0,
        0.6931471805599453,
        1.791759469228055,
        3.1780538303479453,
        4.787491742782046,
        6.579251212010101,
        8.525161361065415,
        10.60460290274525,
        12.80182748008147,
        15.104412573075518,
        17.502307845873887,
        19.98721449566189,
        22.552163853123425,
        25.191221182738683,
        27.899271383840894,
        30.671860106080675,
        33.50507345013689,
        36.39544520803305,
        39.339884187199495,
        42.335616460753485,
        45.38013889847691,
        48.47118135183523,
        51.60667556776438,
        54.784729398112326,
        58.003605222980525,
        61.26170176100201,
        64.55753862700634,
        67.88974313718154,
        71.25703896716801,
        74.65823634883017,
        78.09222355331532,
        81.55795945611504,
        85.05446701758153,
        88.5808275421977,
        92.13617560368711,
        95.71969454214322,
        99.33061245478744,
        102.96819861451382,
        106.63176026064347,
    ],
    dtype="float64",
)


def log_of_product_of_powers(bases, exponents):
    """
    Compute the complex logarithm of ∏ᵢ bases[i]**exponents[i] for real bases.

    Returns
    -------
    re_log : float
        ∑ᵢ exponents[i] * log|bases[i]|, the real part of the log.
    arg_over_pi : float
        (∑ᵢ exponents[i] * arg(bases[i])) / π, i.e. the logarithm’s
        imaginary part scaled by π, reduced modulo 2.

    Notes
    -----
    - If bases[i] == 0 and exponents[i] > 0, contributes −∞ to re_log.
    - If bases[i] == 0 and exponents[i] < 0, re_log accumulates +∞ (ValueError
      may be preferred if you want to reject 0**negative).
    - For bases[i] < 0, arg(bases[i]) = π so each negative base flips
      sign according to the parity of the exponent.
    """
    re_log = 0.0
    arg_over_pi = 0.0

    for b, p in zip(bases, exponents):
        if b == 0:
            if p != 0:
                re_log += -p * float("inf")
        elif b < 0:
            re_log += p * math.log(-b)
            arg_over_pi += p
        else:
            re_log += p * math.log(b)

    return re_log, (arg_over_pi % 2)


@njit
def factorial_lookup(n):
    """
    n! for n<=20
    """
    return factorial_table[n]


@njit
def log_factorial_lookup(n):
    """
    n! for n<=29
    """
    return log_factorial_table[n]


@njit
def factorial(n):
    """ """
    if n < 21:
        return factorial_lookup(n)
    else:
        facn = factorial_lookup(20)
        for _ in range(21, n + 1):
            facn *= _
        return facn


@njit
def log_factorial(n):
    if n < 30:
        return log_factorial_lookup(n)
    result = log_factorial_lookup(29)
    for i in range(30, n + 1):
        result += np.log(i)
    return result


@njit
def integer_power_of_i(power):
    """Compute i^power"""
    power = power % 4
    if power == 0:
        return 1.0 + 0j
    elif power == 1:
        return 0.0 + 1j
    elif power == 2:
        return -1.0 + 0j
    else:  # power == 3
        return 0.0 - 1j


@njit
def Ylm_alt0(l, m, theta, phi):
    """
    factorials make this unstable for large l
    """
    B = (
        1j ** (m + abs(m))
        * np.sqrt((2 * l + 1) * factorial(l + m) * factorial(l - m) + 0.0)
        / (2 ** (abs(m) + 1) * np.sqrt(np.pi))
    )

    val = np.exp(1j * m * phi)
    val_sum = 0.0 * theta + 0j
    for k in range(1 + int((l - abs(m)) / 2)):
        val_sum += (
            (
                (-1) ** k
                * B
                / (
                    4**k
                    * factorial(l - abs(m) - 2 * k)
                    * factorial(k + abs(m))
                    * factorial(k)
                )
            )
            * np.cos(theta) ** (l - abs(m) - 2 * k)
            * np.sin(theta) ** (abs(m) + 2 * k)
        )
    return val * val_sum


@njit
def Ylm_alt1(l, m, theta, phi):
    """
    no safeguards for sin/cos of theta/phi
    Spherical harmonic Ylm(l, m) at angles theta, phi. Works for arrays of angles.

    Args:
        l (int): 0,1,2,...
        m (int): -l,...,l
        theta (float): polar angle in radians
        phi (float): azimuthal angle in radians

    Returns:
        complex: value of spherical harmonic Ylm at angles theta, phi
    """
    abs_m = abs(m)
    exp_itheta = np.exp(1j * theta)
    cos_theta = exp_itheta.real
    sin_theta = exp_itheta.imag
    cos_theta_sqr = cos_theta * cos_theta
    sin_theta_sqr = sin_theta * sin_theta
    q1 = cos_theta ** (l - abs_m) * sin_theta ** (abs_m)
    log_q2 = (
        0.5 * log_factorial(l + abs_m)
        - 0.5 * log_factorial(l - abs_m)
        - log_factorial(abs_m)
    )
    val = q1 * np.exp(log_q2)
    for k in range(1, 1 + int((l - abs_m) / 2)):
        q1 *= -sin_theta_sqr / (4 * cos_theta_sqr)
        log_q2 += (
            np.log(l - abs_m - 2 * k + 2)
            + np.log(l - abs_m - 2 * k + 1)
            - np.log(k + abs_m)
            - np.log(k)
        )
        val += q1 * np.exp(log_q2)
    return (
        val
        * np.exp(1j * m * phi)
        * (
            integer_power_of_i(m + abs_m)
            * np.sqrt((2 * l + 1))
            / (2 ** (abs_m + 1) * np.sqrt(np.pi))
        )
    )


@njit("f8(i8, i8, f8)")
def phi_independent_problem_angle_small_scalarYlm(l, m, theta):
    """
    phi independent part of spherical harmonic Ylm(l, m).
    works for small/zero cos(theta) or sin(theta)

    Args:
        l (int): 0,1,2,...
        m (int): -l,...,l
        theta (float): polar angle in radians [0, pi]

    Returns:
        real: value of Ylm(l, m, theta, phi)/exp(1j*m*phi)
    """

    abs_m = abs(m)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rlm = 1 / (2 * np.sqrt(np.pi))
    if m > 0:
        if m % 2 == 1:
            rlm *= -1
    log_qk = (
        0.5 * np.log(2 * l + 1)
        + 0.5 * log_factorial(l + abs_m)
        - abs_m * np.log(2)
        - 0.5 * log_factorial(l - abs_m)
        - log_factorial(abs_m)
    )
    trig_term = cos_theta ** (l - abs_m) * sin_theta**abs_m
    sk = 1
    sum_Qk = sk * np.exp(log_qk) * trig_term
    minus_log4 = np.log(0.25)
    for k in range(1, 1 + int((l - abs_m) / 2)):
        sk *= -1
        log_qk += (
            minus_log4
            + np.log(l - abs_m - 2 * k + 2)
            + np.log(l - abs_m - 2 * k + 1)
            - np.log(abs_m + k)
            - np.log(k)
        )
        trig_term = cos_theta ** (l - abs_m - 2 * k) * sin_theta ** (abs_m + 2 * k)
        sum_Qk += sk * np.exp(log_qk) * trig_term
    return rlm * sum_Qk


@njit("f8(i8, i8, f8)")
def phi_independent_problem_angle_same_scalarYlm(l, m, theta):
    """
    phi independent part of spherical harmonic Ylm(l, m).
    works for cos(theta)~sin(theta) or cos(theta)=sin(theta)

    Args:
        l (int): 0,1,2,...
        m (int): -l,...,l
        theta (float): polar angle in radians [0, pi]

    Returns:
        real: value of Ylm(l, m, theta, phi)/exp(1j*m*phi)
    """
    # # assumes theta = pi/4 or 3*pi/4 exactly
    # cos_theta = np.cos(theta)
    # tan_theta = np.tan(theta)
    # abs_m = abs(m)
    # rlm = np.sign(cos_theta**l * tan_theta**abs_m) / (2 * np.sqrt(np.pi))
    # if m > 0:
    #     if m % 2 == 1:
    #         rlm *= -1
    # log_qk = (
    #     0.5 * np.log(2 * l + 1)
    #     + 0.5 * log_factorial(l + abs_m)
    #     - 0.5 * l * np.log(2.0)
    #     - abs_m * np.log(2)
    #     - 0.5 * log_factorial(l - abs_m)
    #     - log_factorial(abs_m)
    # )
    # sk = 1
    # sum_Qk = sk * np.exp(log_qk)
    # minus_log4 = np.log(0.25)
    # for k in range(1, 1 + int((l - abs_m) / 2)):
    #     sk *= -1
    #     log_qk += (
    #         minus_log4
    #         + np.log(l - abs_m - 2 * k + 2)
    #         + np.log(l - abs_m - 2 * k + 1)
    #         - np.log(abs_m + k)
    #         - np.log(k)
    #     )
    #     sum_Qk += sk * np.exp(log_qk)
    # return rlm * sum_Qk
    #
    #
    #
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    abs_cos_theta = abs(cos_theta)
    abs_tan_theta = abs(tan_theta)
    log_abs_cos_theta = np.log(abs_cos_theta)
    log_abs_tan_theta = np.log(abs_tan_theta)
    abs_m = abs(m)
    rlm = np.sign(cos_theta**l * tan_theta**abs_m) / (2 * np.sqrt(np.pi))
    if m > 0:
        if m % 2 == 1:
            rlm *= -1
    log_qk = (
        0.5 * np.log(2 * l + 1)
        + 0.5 * log_factorial(l + abs_m)
        + l * log_abs_cos_theta
        + abs_m * log_abs_tan_theta
        - abs_m * np.log(2)
        - 0.5 * log_factorial(l - abs_m)
        - log_factorial(abs_m)
    )
    sk = 1
    sum_Qk = sk * np.exp(log_qk)
    minus_log4 = np.log(0.25)
    for k in range(1, 1 + int((l - abs_m) / 2)):
        sk *= -1
        log_qk += (
            2 * log_abs_tan_theta
            + minus_log4
            + np.log(l - abs_m - 2 * k + 2)
            + np.log(l - abs_m - 2 * k + 1)
            - np.log(abs_m + k)
            - np.log(k)
        )
        sum_Qk += sk * np.exp(log_qk)
    return rlm * sum_Qk


@njit("f8(i8, i8, f8)")
def phi_independent_scalarYlm(l, m, theta):
    """
    phi independent part of spherical harmonic Ylm(l, m)
    Args:
        l (int): 0,1,2,...
        m (int): -l,...,l
        theta (float): polar angle in radians [0, pi]

    Returns:
        real: value of Ylm(l, m, theta, phi)/exp(1j*m*phi)
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    abs_cos_theta = abs(cos_theta)
    abs_sin_theta = abs(sin_theta)
    epsilon = 1e-8
    if abs_cos_theta < epsilon:
        return phi_independent_problem_angle_small_scalarYlm(l, m, theta)
    if abs_sin_theta < epsilon:
        return phi_independent_problem_angle_small_scalarYlm(l, m, theta)
    if abs(abs_cos_theta - abs_sin_theta) < epsilon:
        return phi_independent_problem_angle_same_scalarYlm(l, m, theta)
    log_abs_cos_theta = np.log(abs_cos_theta)
    log_abs_sin_theta = np.log(abs_sin_theta)
    abs_m = abs(m)
    rlm = np.sign(cos_theta ** (l - abs_m) * sin_theta**abs_m) / (2 * np.sqrt(np.pi))
    if m > 0:
        if m % 2 == 1:
            rlm *= -1
    # ep, en, op, on
    # +, +, -, +
    #
    #
    #
    # if m % 2 == 1:
    #     rlm *= -1
    # ep,en,op,on
    # +, +, -, -
    #
    #
    #
    #
    #
    # if m < 0:
    #     if m % 2 == 1:
    #         rlm *= -1
    # ep,en,op,on
    # +, +, +, -
    #
    #
    #
    #
    log_qk = (
        0.5 * np.log(2 * l + 1)
        + 0.5 * log_factorial(l + abs_m)
        + (l - abs_m) * log_abs_cos_theta
        + abs_m * log_abs_sin_theta
        - abs_m * np.log(2)
        - 0.5 * log_factorial(l - abs_m)
        - log_factorial(abs_m)
    )
    sk = 1
    sum_Qk = sk * np.exp(log_qk)
    minus_log4 = np.log(0.25)
    for k in range(1, 1 + int((l - abs_m) / 2)):
        sk *= -1
        log_qk += (
            -2 * log_abs_cos_theta
            + 2 * log_abs_sin_theta
            + minus_log4
            + np.log(l - abs_m - 2 * k + 2)
            + np.log(l - abs_m - 2 * k + 1)
            - np.log(abs_m + k)
            - np.log(k)
        )
        sum_Qk += sk * np.exp(log_qk)
    return rlm * sum_Qk


@njit("c16(i8, i8, f8, f8)")
def scalarYlm(l, m, theta, phi):
    """
    Spherical harmonic Ylm(l, m)
    Args:
        l (int): 0,1,2,...
        m (int): -l,...,l
        theta (float): polar angle in radians
        phi (float): polar angle in radians

    Returns:
        complex: value of Ylm(l, m, theta, phi)
    """
    return np.exp(1j * m * phi) * phi_independent_scalarYlm(l, m, theta)


@njit("c16[:](i8, i8, f8[:], f8[:])")
def Ylm(l, m, theta, phi):
    """
    Vectorized spherical harmonic Ylm(l, m)

    Args:
        l (int): 0,1,2,...
        m (int): -l,...,l
        theta (array): polar angle in radians
        phi (array): azimuthal angle in radians


    Returns:
        array: values of spherical harmonic Ylm at angles theta, phi
    """
    Ns = len(theta)
    Y = np.zeros(Ns)
    for s in range(Ns):
        Y[s] = phi_independent_scalarYlm(l, m, theta[s])
    return np.exp(1j * m * phi) * Y


@njit("f8[:](i8, i8, f8[:], f8[:])")
def real_Ylm(l, m, theta, phi):
    """
    Real form of spherical harmonic.
    """
    abs_m = abs(m)
    Ns = len(theta)
    Y = np.zeros(Ns)
    for s in range(Ns):
        Y[s] = phi_independent_scalarYlm(l, abs_m, theta[s])
    if m < 0:
        return np.sqrt(2) * (-1) ** abs_m * np.sin(abs_m * phi) * Y
    elif m == 0:
        return Y
    else:  # m >0
        return np.sqrt(2) * (-1) ** abs_m * np.cos(abs_m * phi) * Y


def build_sh_module():
    from numba.pycc import CC

    @njit
    def jit_n_LM(l, m):
        """
        Enumeration of spherical harmonic indices.

        (0, 0) -> 0
        (1, -1) -> 1
        (1, 0) -> 2
        (1, 1) -> 3
        (2, -2) -> 4
        .
        .
        .
        """
        return m + l**2 + l

    @njit
    def jit_lm_N(n):
        """
        get (l, m) spherical harmonic indices from n

        0 -> (0, 0)
        1 -> (1, -1)
        2 -> (1, 0)
        3 -> (1, 1)
        4 -> (2, -2)
        .
        .
        .
        """
        l = int(np.sqrt(n))
        m = n - (l**2 + l)
        return l, m

    cc = CC("spherical_harmonics")
    cc.output_dir = "./build/pybrane"
    cc.verbose = True

    @cc.export("n_LM", "i8(i8, i8)")
    def n_LM(l, m):
        return m + l**2 + l

    @cc.export("lm_N", "i8[:](i8)")
    def lm_N(n):
        lm = np.zeros(2, dtype=np.int64)
        lm[0] = int(np.sqrt(n))
        lm[1] = n - (lm[0] ** 2 + lm[0])
        return lm

    @cc.export("Ylm", "c16[:](i8, i8, f8[:], f8[:])")
    def _Ylm(l, m, theta, phi):
        return Ylm(l, m, theta, phi)

    @cc.export("real_Ylm", "f8[:](i8, i8, f8[:], f8[:])")
    def _real_Ylm(l, m, theta, phi):
        return real_Ylm(l, m, theta, phi)

    cc.compile()
