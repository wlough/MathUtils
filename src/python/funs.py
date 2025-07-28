import numpy as np
from numba import njit


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
def _log_factorial(n):
    if n <= 1:
        return 0.0
    result = 0.0
    for i in range(2, n + 1):
        result += np.log(i)
    return result


@njit
def log_factorial(n):
    if n < 30:
        return log_factorial_lookup(n)
    result = log_factorial_lookup(29)
    for i in range(30, n + 1):
        result += np.log(i)
    return result
