import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.sparse import diags_array


# ####################
# Misc functions
def linspace_nu(start=0, stop=1, num=100, jitter=0.1, endpoint=True, rng=None):
    # seed = 0
    rng = rng if rng is not None else np.random.default_rng()
    if endpoint:

        x_u = np.linspace(start, stop, num)
        x = x_u.copy()
        mean_dx = np.ptp(x) / num
        x[1:-1] += jitter * mean_dx * rng.standard_normal(num - 2)

        x = np.clip(x, start, stop)
        x.sort()
        i = 0
        while abs(x[i] - x[0]) < 1e-10:
            i += 1
        if i > 1:
            print(f"resampling boundaries samples {i} from start")
            x[:i] = np.linspace(x[0], x[i], i)

        i = 0
        while abs(x[-(i + 1)] - x[-1]) < 1e-10:
            i += 1
        if i > 1:
            print(f"resampling boundaries samples {i} from stop")
            # print(x[-(i+1):])
            # print(np.linspace(x[-(i+1)], x[-1], i+1))
            x[-(i + 1) :] = np.linspace(x[-(i + 1)], x[-1], i + 1)

        return x
    else:

        x_u = np.linspace(start, stop, num, endpoint=endpoint)
        x = x_u.copy()
        mean_dx = np.ptp(x) / num
        x[1:-1] += jitter * mean_dx * rng.standard_normal(num - 2)

        x = np.clip(x, start, stop)
        x.sort()
        i = 0
        while abs(x[i] - x[0]) < 1e-10:
            i += 1
        if i > 1:
            print(f"resampling boundaries samples {i} from start")
            x[:i] = np.linspace(x[0], x[i], i)

        i = 0
        while abs(x[-(i + 1)] - x[-1]) < 1e-10:
            i += 1
        if i > 1:
            print(f"resampling boundaries samples {i} from stop")
            # print(x[-(i+1):])
            # print(np.linspace(x[-(i+1)], x[-1], i+1))
            x[-(i + 1) :] = np.linspace(x[-(i + 1)], x[-1], i + 1)

        return x


######################
######################
# Finite differences #
######################
######################
def trapint(y, x, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)

    int_f = 0.5 * (x[1 % x.size] - x[0]) * f[0] + 0.5 * (x[-1] - x[-2 % x.size]) * f[-1]
    for j in range(1, len(f) - 1):
        int_f += 0.5 * (x[j + 1] - x[j - 1]) * f[j]
    return int_f


def trapint_dx(y, dx, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    int_f = 0.5 * (dx) * f[0] + 0.5 * (dx) * f[-1]
    for j in range(1, len(f) - 1):
        int_f += 0.5 * (2 * dx) * f[j]
    return int_f


def cumtrapint_dx(y, dx, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    int_f = np.zeros_like(f)
    for j in range(1, len(f)):
        int_f[j] = int_f[j - 1] + 0.5 * dx * (f[j] + f[j - 1])
    return np.moveaxis(int_f, 0, axis)


def cumtrapint(y, x, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    int_f = np.zeros_like(f)
    for j in range(1, len(f)):
        int_f[j] = int_f[j - 1] + 0.5 * (x[j] - x[j - 1]) * (f[j] + f[j - 1])
    return np.moveaxis(int_f, 0, axis)


def diff(y, dx, n, axis=0):
    """
    2nd order accurate 0th to 4th finite difference on
    a uniform grid with spacing dx

    Arguments
    ---------
    y : array_like
        Function samples.
    dx : float
        Grid spacing.
    n : int
        Order of the derivative.
    axis : int
        Axis along which difference is taken.

    Returns
    -------
    ndarray
        Finite difference of input array
    """
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    if n == 0:
        return np.moveaxis(f, 0, axis)
    elif n == 1:
        df = np.zeros(f.shape)
        df[0] = (-1.5 * f[0] + 2.0 * f[1] - 0.5 * f[2]) / dx
        df[-1] = (0.5 * f[-3] - 2.0 * f[-2] + 1.5 * f[-1]) / dx
        df[1:-1] = (-0.5 * f[:-2] + 0.5 * f[2:]) / dx
        return np.moveaxis(df, 0, axis)
    elif n == 2:
        df = np.zeros(f.shape)
        df[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / dx**2
        df[-1] = (-f[-4] + 4.0 * f[-3] - 5.0 * f[-2] + 2.0 * f[-1]) / dx**2

        df[1:-1] = (f[:-2] - 2 * f[1:-1] + f[2:]) / dx**2
        return np.moveaxis(df, 0, axis)
    elif n == 3:
        df = np.zeros(f.shape)
        df[0] = (
            -2.5 * f[0] + 9.0 * f[1] - 12.0 * f[2] + 7.0 * f[3] - 1.5 * f[4]
        ) / dx**3
        df[1] = (
            -1.5 * f[0] + 5.0 * f[1] - 6.0 * f[2] + 3.0 * f[3] - 0.5 * f[4]
        ) / dx**3
        df[-2] = (
            0.5 * f[-5] - 3.0 * f[-4] + 6.0 * f[-3] - 5.0 * f[-2] + 1.5 * f[-1]
        ) / dx**3
        df[-1] = (
            1.5 * f[-5] - 7.0 * f[-4] + 12.0 * f[-3] - 9.0 * f[-2] + 2.5 * f[-1]
        ) / dx**3

        df[2:-2] = (-0.5 * f[:-4] + f[1:-3] - f[3:-1] + 0.5 * f[4:]) / dx**3
        return np.moveaxis(df, 0, axis)
    elif n == 4:
        df = np.zeros(f.shape)
        df[0] = (
            3 * f[0] - 14 * f[1] + 26 * f[2] - 24 * f[3] + 11 * f[4] - 2 * f[5]
        ) / dx**4
        df[1] = (2 * f[0] - 9 * f[1] + 16 * f[2] - 14 * f[3] + 6 * f[4] - f[5]) / dx**4
        df[-2] = (
            -f[-6] + 6 * f[-5] - 14 * f[-4] + 16 * f[-3] - 9 * f[-2] + 2 * f[-1]
        ) / dx**4
        df[-1] = (
            -2 * f[-6] + 11 * f[-5] - 24 * f[-4] + 26 * f[-3] - 14 * f[-2] + 3 * f[-1]
        ) / dx**4
        df[2:-2] = (f[:-4] - 4 * f[1:-3] + 6 * f[2:-2] - 4 * f[3:-1] + f[4:]) / dx**4

        return np.moveaxis(df, 0, axis)
    raise ValueError(
        f"Only derivatives 1 through 4 are implemented, but {n=} was given"
    )


def rdiff1(y, dx, axis=0, periodic=False):

    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    df = (0.5 * np.roll(f, -1, axis=0) - 0.5 * np.roll(f, 1, axis=0)) / dx
    if periodic:
        return np.moveaxis(df, 0, axis)
    df[0] = (-1.5 * f[0] + 2.0 * f[1] - 0.5 * f[2]) / dx
    df[-1] = (0.5 * f[-3] - 2.0 * f[-2] + 1.5 * f[-1]) / dx
    return np.moveaxis(df, 0, axis)


def rdiff2(y, dx, axis=0, periodic=False):
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    df = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2
    if periodic:
        return np.moveaxis(df, 0, axis)
    df[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / dx**2
    df[-1] = (-f[-4] + 4.0 * f[-3] - 5.0 * f[-2] + 2.0 * f[-1]) / dx**2
    return np.moveaxis(df, 0, axis)


def rdiff3(y, dx, axis=0, periodic=False):
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    df = (
        0.5 * np.roll(f, -2, axis=0)
        - np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=0)
        - 0.5 * np.roll(f, 2, axis=0)
    ) / dx**3
    if periodic:
        return np.moveaxis(df, 0, axis)
    df[0] = (-2.5 * f[0] + 9.0 * f[1] - 12.0 * f[2] + 7.0 * f[3] - 1.5 * f[4]) / dx**3
    df[1] = (-1.5 * f[0] + 5.0 * f[1] - 6.0 * f[2] + 3.0 * f[3] - 0.5 * f[4]) / dx**3
    df[-2] = (
        0.5 * f[-5] - 3.0 * f[-4] + 6.0 * f[-3] - 5.0 * f[-2] + 1.5 * f[-1]
    ) / dx**3
    df[-1] = (
        1.5 * f[-5] - 7.0 * f[-4] + 12.0 * f[-3] - 9.0 * f[-2] + 2.5 * f[-1]
    ) / dx**3
    return np.moveaxis(df, 0, axis)


def rdiff4(y, dx, axis=0, periodic=False):
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)
    df = (
        np.roll(f, -2, axis=0)
        - 4 * np.roll(f, -1, axis=0)
        + 6 * f
        - 4 * np.roll(f, 1, axis=0)
        + np.roll(f, 2, axis=0)
    ) / dx**4
    if periodic:
        return np.moveaxis(df, 0, axis)
    df[0] = (
        3 * f[0] - 14 * f[1] + 26 * f[2] - 24 * f[3] + 11 * f[4] - 2 * f[5]
    ) / dx**4
    df[1] = (2 * f[0] - 9 * f[1] + 16 * f[2] - 14 * f[3] + 6 * f[4] - f[5]) / dx**4
    df[-2] = (
        -f[-6] + 6 * f[-5] - 14 * f[-4] + 16 * f[-3] - 9 * f[-2] + 2 * f[-1]
    ) / dx**4
    df[-1] = (
        -2 * f[-6] + 11 * f[-5] - 24 * f[-4] + 26 * f[-3] - 14 * f[-2] + 3 * f[-1]
    ) / dx**4
    return np.moveaxis(df, 0, axis)


def sym_findiff_weights(deriv_order, sample_number):
    """
    Makes sympy expressions of weights for finite difference of order
    'deriv_order' from 'sample_number' samples.

    derivative Df(x) ~= (d/dx)^{deriv_order}f is at x
    using points [x+h[0],...,x+h[j],...]


    h1, a1 = sym_findiff_weights(1, 3)
    h2, a2 = sym_findiff_weights(2, 4)
    h3, a3 = sym_findiff_weights(3, 5)
    h4, a4 = sym_findiff_weights(4, 6)
    a1.simplify()
    a2.simplify()
    a3.simplify()
    a4.applyfunc(lambda _: _.expand().simplify().factor())
    """
    h = sp.Matrix([sp.symbols(f"h_{j}") for j in range(sample_number)])
    weights = sp.Array(sp.finite_diff_weights(deriv_order, h, x0=0))
    a = weights[deriv_order, -1].simplify()
    return h, a


def findiff_weights1(h):
    h0, h1, h2 = h
    return np.array(
        [
            -(h1 + h2) / ((h1 - h0) * (h2 - h0)),
            -(h0 + h2) / ((h2 - h1) * (h0 - h1)),
            -(h0 + h1) / ((h0 - h2) * (h1 - h2)),
        ]
    )


def findiff_weights2(h):
    h0, h1, h2, h3 = h
    return np.array(
        [
            2 * (h1 + h2 + h3) / ((h1 - h0) * (h2 - h0) * (h3 - h0)),
            2 * (h0 + h2 + h3) / ((h0 - h1) * (h2 - h1) * (h3 - h1)),
            2 * (h0 + h1 + h3) / ((h0 - h2) * (h1 - h2) * (h3 - h2)),
            2 * (h0 + h1 + h2) / ((h0 - h3) * (h1 - h3) * (h2 - h3)),
        ]
    )


def findiff_weights3(h):
    h0, h1, h2, h3, h4 = h
    return np.array(
        [
            -6 * (h1 + h2 + h3 + h4) / ((h1 - h0) * (h2 - h0) * (h3 - h0) * (h4 - h0)),
            -6 * (h0 + h2 + h3 + h4) / ((h0 - h1) * (h2 - h1) * (h3 - h1) * (h4 - h1)),
            -6 * (h0 + h1 + h3 + h4) / ((h0 - h2) * (h1 - h2) * (h3 - h2) * (h4 - h2)),
            -6 * (h0 + h1 + h2 + h4) / ((h0 - h3) * (h1 - h3) * (h2 - h3) * (h4 - h3)),
            -6 * (h0 + h1 + h2 + h3) / ((h0 - h4) * (h1 - h4) * (h2 - h4) * (h3 - h4)),
        ]
    )


def findiff_weights4(h):
    h0, h1, h2, h3, h4, h5 = h
    return np.array(
        [
            24
            * (h1 + h2 + h3 + h4 + h5)
            / ((h1 - h0) * (h2 - h0) * (h3 - h0) * (h4 - h0) * (h5 - h0)),
            24
            * (h0 + h2 + h3 + h4 + h5)
            / ((h0 - h1) * (h2 - h1) * (h3 - h1) * (h4 - h1) * (h5 - h1)),
            24
            * (h0 + h1 + h3 + h4 + h5)
            / ((h0 - h2) * (h1 - h2) * (h3 - h2) * (h4 - h2) * (h5 - h2)),
            24
            * (h0 + h1 + h2 + h4 + h5)
            / ((h0 - h3) * (h1 - h3) * (h2 - h3) * (h4 - h3) * (h5 - h3)),
            24
            * (h0 + h1 + h2 + h3 + h5)
            / ((h0 - h4) * (h1 - h4) * (h2 - h4) * (h3 - h4) * (h5 - h4)),
            24
            * (h0 + h1 + h2 + h3 + h4)
            / ((h0 - h5) * (h1 - h5) * (h2 - h5) * (h3 - h5) * (h4 - h5)),
        ]
    )


def diff_nu(y, x, deriv_order=1, axis=0):
    """
    2nd order accurate 1st to 4th finite difference on
    a possibly nonuniform grid.

    Arguments
    ---------
    y : array_like
        Function samples.
    x : array_like
        Grid points
    deriv_order : int
        Order of the derivative.
    axis : int
        Axis along which the difference is taken.

    Returns
    -------
    ndarray
        Finite difference of y at each grid point in x
    """
    y = np.asarray(y)
    f = np.moveaxis(y, axis, 0)

    if deriv_order == 1:
        df = np.zeros(f.shape)
        h = x[:3] - x[0]
        a = findiff_weights1(h)
        df[0] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2]
        for j in range(1, len(f) - 1):
            h = x[j - 1 : j + 2] - x[j]
            a = findiff_weights1(h)
            df[j] = (
                a[0] * f[j - 1] + a[1] * f[j] + a[2] * f[j + 1]
            )  # np.einsum('i,i...', a,f[j-1:j+2])
        h = x[-3:] - x[-1]
        a = findiff_weights1(h)
        df[-1] = (
            a[0] * f[-3] + a[1] * f[-2] + a[2] * f[-1]
        )  # np.einsum('i,i...', a,f[-3:])
        return np.moveaxis(df, 0, axis)

    elif deriv_order == 2:
        df = np.zeros(f.shape)
        h = x[:4] - x[0]
        a = findiff_weights2(h)
        df[0] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3]

        h = x[:4] - x[1]
        a = findiff_weights2(h)
        df[1] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3]
        for j in range(2, len(f) - 1):
            h = x[j - 2 : j + 2] - x[j]
            a = findiff_weights2(h)
            df[j] = a[0] * f[j - 2] + a[1] * f[j - 1] + a[2] * f[j] + a[3] * f[j + 1]
        h = x[-4:] - x[-1]
        a = findiff_weights2(h)
        df[-1] = a[0] * f[-4] + a[1] * f[-3] + a[2] * f[-2] + a[3] * f[-1]
        return np.moveaxis(df, 0, axis)

    elif deriv_order == 3:
        df = np.zeros(f.shape)
        h = x[:5] - x[0]
        a = findiff_weights3(h)
        df[0] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3] + a[4] * f[4]

        h = x[:5] - x[1]
        a = findiff_weights3(h)
        df[1] = a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + a[3] * f[3] + a[4] * f[4]

        for j in range(2, len(f) - 2):
            h = x[j - 2 : j + 3] - x[j]
            a = findiff_weights3(h)
            df[j] = (
                a[0] * f[j - 2]
                + a[1] * f[j - 1]
                + a[2] * f[j]
                + a[3] * f[j + 1]
                + a[4] * f[j + 2]
            )

        h = x[-5:] - x[-2]
        a = findiff_weights3(h)
        df[-2] = (
            a[0] * f[-5] + a[1] * f[-4] + a[2] * f[-3] + a[3] * f[-2] + a[4] * f[-1]
        )

        h = x[-5:] - x[-1]
        a = findiff_weights3(h)
        df[-1] = (
            a[0] * f[-5] + a[1] * f[-4] + a[2] * f[-3] + a[3] * f[-2] + a[4] * f[-1]
        )
        return np.moveaxis(df, 0, axis)
    elif deriv_order == 4:
        df = np.zeros(f.shape)
        h = x[:6] - x[0]
        a = findiff_weights4(h)
        df[0] = (
            a[0] * f[0]
            + a[1] * f[1]
            + a[2] * f[2]
            + a[3] * f[3]
            + a[4] * f[4]
            + a[5] * f[5]
        )

        h = x[:6] - x[1]
        a = findiff_weights4(h)
        df[1] = (
            a[0] * f[0]
            + a[1] * f[1]
            + a[2] * f[2]
            + a[3] * f[3]
            + a[4] * f[4]
            + a[5] * f[5]
        )

        h = x[:6] - x[2]
        a = findiff_weights4(h)
        df[2] = (
            a[0] * f[0]
            + a[1] * f[1]
            + a[2] * f[2]
            + a[3] * f[3]
            + a[4] * f[4]
            + a[5] * f[5]
        )

        for j in range(3, len(f) - 2):
            h = x[j - 3 : j + 3] - x[j]
            a = findiff_weights4(h)
            df[j] = (
                a[0] * f[j - 3]
                + a[1] * f[j - 2]
                + a[2] * f[j - 1]
                + a[3] * f[j]
                + a[4] * f[j + 1]
                + a[5] * f[j + 2]
            )

        h = x[-6:] - x[-2]
        a = findiff_weights4(h)
        df[-2] = (
            a[0] * f[-6]
            + a[1] * f[-5]
            + a[2] * f[-4]
            + a[3] * f[-3]
            + a[4] * f[-2]
            + a[5] * f[-1]
        )

        h = x[-6:] - x[-1]
        a = findiff_weights4(h)
        df[-1] = (
            a[0] * f[-6]
            + a[1] * f[-5]
            + a[2] * f[-4]
            + a[3] * f[-3]
            + a[4] * f[-2]
            + a[5] * f[-1]
        )
        return np.moveaxis(df, 0, axis)
    raise ValueError(
        f"Only derivatives 1 through 4 are implemented, but {deriv_order=} was given"
    )


def rdiff(y, dx, deriv_order=1, axis=0, periodic=True):
    """
    2nd order accurate 1st to 4th finite difference on
    a uniform grid with spacing dx

    Arguments
    ---------
    y : array_like
        Function samples.
    dx : float
        Grid spacing.
    deriv_order : int
        Order of the derivative.
    axis : int
        Axis along which difference is taken.

    Returns
    -------
    ndarray
        Finite difference of input array
    """
    if deriv_order == 1:
        return rdiff1(y, dx, axis=axis, periodic=periodic)
    if deriv_order == 2:
        return rdiff2(y, dx, axis=axis, periodic=periodic)
    if deriv_order == 3:
        return rdiff3(y, dx, axis=axis, periodic=periodic)
    if deriv_order == 4:
        return rdiff4(y, dx, axis=axis, periodic=periodic)
    raise ValueError(
        f"Only derivatives 1 through 4 are implemented, but {deriv_order=} was given"
    )


def minimal_stencil_width(diff_order, accuracy_order, centered=False):
    """
    Minimal number of points (width w) needed to achieve at least `accuracy_order`
    for the m-th derivative.

    Parameters
    ----------
    diff_order : int  # m
        Order of the derivative.
    accuracy_order : int  # p
        Desired order of accuracy (truncation error O(h**p)).
    centered : bool
        Use a symmetric (centered) stencil if True.

    Returns
    -------
    int
        Minimal stencil width w (number of points).
    """
    m = int(diff_order)
    p = int(accuracy_order)
    if m < 0 or p < 1:
        raise ValueError(
            f"diff_order >= 0 and accuracy_order >= 1 required, but {diff_order=} and {accuracy_order=} were given."
        )
    if centered:
        # enforce even p for a centered stencil
        if p % 2 == 1:
            p += 1
        # symmetry bonus for even m
        w = m + p - (1 if (m % 2 == 0) else 0)
        # centered stencils have odd width
        if w % 2 == 0:
            w += 1
        return w
    else:
        # generic minimal count; biasing handled elsewhere
        return m + p


def is_uniform_grid(x, test_frac=0.1, tol=1e-6, rng=None):
    """
    Test if grid points are evenly spaced by random sampling.

    Arguments
    ---------
    x : array_like
        Grid points.
    test_frac : float
        Fraction of grid to check
    tol : float
        Tolerance
    rng : Generator
        Random number generator
    Returns
    -------
    bool
        Whether or not grid is spacing is uniform to within tolerance
    """
    x = np.asarray(x)
    Nx = x.size
    if Nx < 2:
        raise ValueError("x must have at least 2 points.")
    Npairs = Nx - 1
    Ntest = max(int(round(Nx * test_frac)), 2)
    Ntest = min(Ntest, Npairs)
    rng = np.random.default_rng() if rng is None else rng
    I_test = rng.choice(Npairs, size=Ntest, replace=False)
    dx = x[I_test + 1] - x[I_test]
    dx_min, dx_max = dx.min(), dx.max()
    if dx_min < 0:
        raise ValueError("x must be sorted.")
    return (dx_max - dx_min) / dx.mean() < tol


def fornberg_weights(x_stencil, x0, d_max):
    """
    Weights for derivatives 0 to d_max at x0 on nodes x_stencil using Fornberg's algorithm. Returns W with shape (d_max+1, n), where W[d,j] is weight for f(x_j) for d-th derivative.

    (D^{d} f)(x0) = Sum_{j}( W[d,j] * f(x_stencil[j]) )

    Arguments
    ---------
    x_stencil : array_like
        Grid points to use.
    x0 : float
        Point where derivative is evaluated.
    d_max : int
        Maximum derivative to calculate weights for.

    Returns
    -------
    ndarray
        Weights for derivatives.
    """
    x = np.asarray(x_stencil, dtype=float)
    n = x.size
    d_max = int(d_max)
    W = np.zeros((d_max + 1, n), dtype=float)
    c1 = 1.0
    c4 = x[0] - x0
    W[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, d_max)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    W[k, i] = (c1 * (k * W[k - 1, i - 1] - c5 * W[k, i - 1])) / c2
                W[0, i] = (-c1 * c5 * W[0, i - 1]) / c2
            for k in range(mn, 0, -1):
                W[k, j] = (c4 * W[k, j] - k * W[k - 1, j]) / c3
            W[0, j] = (c4 * W[0, j]) / c3
        c1 = c2
    return W


def get_fornberg_data(
    x, stencil, deriv_order, period=None, boundary0_stencil=None, boundary1_stencil=None
):
    """
    Uses Fornberg's algorithm to compute weights for derivative `deriv_order`
    at all points in x[i] in x using stencil points [x[i+s] for s in stencil].

    Arguments
    ---------
    x : array_like
        Grid points.
    stencil : array_like
        Stencil index offsets.
    deriv_order : int
        Order of derivative.
    period : float or None
        Period of x domain.
    boundary0_stencil : array_like or None
        Stencil index offsets at x[0] boundary. Overridden when period is not None
    boundary1_stencil : array_like or None
        Stencil index offsets at x[-1] boundary. Overridden when period is not None

    Returns
    -------
    W : list of ndarray
        W[i] = nonzero elements in row `i` of difference matrix
    K : list of ndarray
        K[i] = column indices of nonzero elements in row `i` of difference matrix
    """
    d_max = int(deriv_order)
    stencil = np.asarray(stencil, dtype=int)
    x = np.asarray(x, dtype=float)
    Nx = len(x)

    min_offset = min(stencil)
    max_offset = max(stencil)

    interior_start = max(0, -min_offset)
    interior_end = min(Nx, Nx - max_offset)
    interior_range = range(interior_start, interior_end)

    boundary0_range = range(0, interior_start)
    boundary1_range = range(interior_end, Nx)

    get_interior_indices = lambda n: stencil + n
    if period is not None:
        boundary0_stencil = stencil
        boundary1_stencil = stencil
        # Tx = period
        get_boundary0_indices = lambda n: (n + boundary0_stencil) % Nx
        get_boundary1_indices = lambda n: (n + boundary1_stencil) % Nx
        get_dx = lambda ind, n: (x[ind] - x[n] + period / 2) % period - period / 2
    else:
        if boundary0_stencil is None:
            boundary0_stencil = stencil - min_offset
        if boundary1_stencil is None:
            boundary1_stencil = stencil - max_offset
        get_boundary0_indices = lambda n: boundary0_stencil
        get_boundary1_indices = lambda n: boundary1_stencil + Nx - 1
        get_dx = lambda ind, n: x[ind] - x[n]

    W = []
    K = []
    for stencil_width, sample_range, get_indices in zip(
        [boundary0_stencil.size, stencil.size, boundary1_stencil.size],
        [boundary0_range, interior_range, boundary1_range],
        [get_boundary0_indices, get_interior_indices, get_boundary1_indices],
    ):
        for n in sample_range:
            Kn = get_indices(n)
            K.append(Kn)
            dx = get_dx(Kn, n)
            Wn = np.zeros((d_max + 1, stencil_width), dtype=float)
            c1 = 1.0
            c4 = dx[0]
            Wn[0, 0] = 1.0
            for i in range(1, stencil_width):
                mn = min(i, d_max)
                c2 = 1.0
                c5 = c4
                c4 = dx[i]
                for j in range(i):
                    c3 = dx[i] - dx[j]
                    c2 *= c3
                    if j == i - 1:
                        for k in range(mn, 0, -1):
                            Wn[k, i] = (
                                c1 * (k * Wn[k - 1, i - 1] - c5 * Wn[k, i - 1])
                            ) / c2
                        Wn[0, i] = (-c1 * c5 * Wn[0, i - 1]) / c2
                    for k in range(mn, 0, -1):
                        Wn[k, j] = (c4 * Wn[k, j] - k * Wn[k - 1, j]) / c3
                    Wn[0, j] = (c4 * Wn[0, j]) / c3
                c1 = c2
            W.append(Wn[-1])

    return W, K


def sparse_arr_from_weights_indices(Wd, K):
    """
    Build sparse finite array from lists of nonzero weights and indices
    """
    Nx = len(Wd)
    diagonal_offsets = [I - i for i, I in enumerate(K)]
    offsets = np.unique(np.concatenate(diagonal_offsets), sorted=True)
    diagonals_padded = np.zeros((offsets.size, Nx), dtype=float)
    dummy_at_offset = {o: _ for _, o in enumerate(offsets)}
    for i in range(Nx):
        dummy_slice = np.array(
            [dummy_at_offset[oi] for oi in diagonal_offsets[i]], dtype=int
        )
        diagonals_padded[dummy_slice, i] = Wd[i]

    diagonals = []
    for diag_padded, offset in zip(diagonals_padded, offsets):
        if offset < 0:
            diagonals.append(diag_padded[-offset:])
        elif offset > 0:
            diagonals.append(diag_padded[:-offset])
        else:
            diagonals.append(diag_padded[:])
    return diags_array(diagonals, offsets=offsets, format="lil")


def get_fornberg_findiff_arr(x, acc, deriv_order, period=None):
    deriv_order = int(deriv_order)
    acc = int(acc)
    if deriv_order < 0 or acc < 1:
        raise ValueError(
            f"deriv_order >= 0 and acc >= 1 required, but {deriv_order=} and {acc=} were given."
        )
    x_is_uniform = is_uniform_grid(x, test_frac=0.1, tol=1e-6)
    stencil_width = deriv_order + acc
    if x_is_uniform and (deriv_order % 2 == 0):
        stencil_width -= 1
    if stencil_width % 2 == 0:
        stencil_width += 1
    stencil_radius = (stencil_width - 1) // 2

    bdry_stencil_width = deriv_order + acc
    if bdry_stencil_width % 2 == 0:
        bdry_stencil_width += 1
    bdry_stencil_radius = (bdry_stencil_width - 1) // 2

    stencil = np.arange(-stencil_radius, stencil_radius + 1)
    boundary0_stencil = np.arange(0, 2 * bdry_stencil_radius + 1)
    boundary1_stencil = np.arange(-2 * bdry_stencil_radius, 1)
    W, K = get_fornberg_data(
        x,
        stencil,
        deriv_order,
        period=period,
        boundary0_stencil=boundary0_stencil,
        boundary1_stencil=boundary1_stencil,
    )

    return sparse_arr_from_weights_indices(W, K)


def get_fornberg_stencils(acc, deriv_order, x_is_uniform=False, periodic=False):
    deriv_order = int(deriv_order)
    acc = int(acc)
    if deriv_order < 0 or acc < 1:
        raise ValueError(
            f"deriv_order >= 0 and acc >= 1 required, but {deriv_order=} and {acc=} were given."
        )
    stencil_width = deriv_order + acc
    if x_is_uniform and (deriv_order % 2 == 0):
        stencil_width -= 1
    if stencil_width % 2 == 0:
        stencil_width += 1
    stencil_radius = (stencil_width - 1) // 2

    bdry_stencil_width = deriv_order + acc
    if bdry_stencil_width % 2 == 0:
        bdry_stencil_width += 1
    bdry_stencil_radius = (bdry_stencil_width - 1) // 2

    stencil = np.arange(-stencil_radius, stencil_radius + 1)
    boundary0_stencil = np.arange(0, 2 * bdry_stencil_radius + 1)
    boundary1_stencil = np.arange(-2 * bdry_stencil_radius, 1)

    return {
        "interior": stencil,
        "boundary0": boundary0_stencil,
        "boundary1": boundary1_stencil,
    }


class FiniteDifferenceOps:
    def __init__(
        self,
        x=None,
        max_deriv_order=4,
        acc=2,
        period=None,
        rng=None,
    ):
        self.x = x
        self.period = period
        self.deriv_orders = list(range(0, max_deriv_order + 1))
        self.acc = acc
        self.fornberg_findiff_mats = self.update_fornberg_findiff_mats()
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def period(self):
        return self.period_

    @period.setter
    def period(self, value):
        self.period_ = np.asarray(value) if value is not None else value

    @property
    def x(self):
        return self.x_

    @x.setter
    def x(self, value):
        self.x_ = np.asarray(value, dtype=float, copy=True)

    @property
    def deriv_orders(self):
        return self.deriv_orders_

    @deriv_orders.setter
    def deriv_orders(self, value):
        self.deriv_orders_ = np.asarray(value, dtype=int, copy=True)

    @property
    def acc(self):
        return self.acc_

    @acc.setter
    def acc(self, value):
        self.acc_ = int(value)

    @property
    def fornberg_findiff_mats(self):
        return self.fornberg_findiff_mats_

    @fornberg_findiff_mats.setter
    def fornberg_findiff_mats(self, value):
        self.fornberg_findiff_mats_ = dict(value)

    def update_fornberg_findiff_mats(self):
        return {
            d: get_fornberg_findiff_arr(self.x, self.acc, d, self.period)
            for d in self.deriv_orders
        }

    def diff(self, y, deriv_order, axis=0):
        if axis == 0:
            return self.fornberg_findiff_mats[deriv_order] @ y
        y = np.asarray(y)
        f = np.moveaxis(y, axis, 0)
        return np.moveaxis(self.fornberg_findiff_mats[deriv_order] @ f, 0, axis)


def diff_fornberg_test(
    min_deriv_order=0, max_deriv_order=4, acc=2, periodic=False, rng=None
):
    rng = np.random.default_rng(0) if rng is None else rng
    # a, b = 0.0, 1.0
    # k = 2 * np.pi
    a, b = 0.0, 2 * np.pi
    k = 1
    _x = lambda s: np.cos(k * s)
    _y = lambda s: np.sin(k * s)
    jitter = 0.1
    Ns = 111
    s = linspace_nu(
        start=a, stop=b, num=Ns, jitter=jitter, endpoint=not periodic, rng=rng
    )
    # print(b-s[-1])
    x, y = _x(s), _y(s)
    r = np.vstack((x, y)).T
    deriv_orders = list(range(0, max_deriv_order + 1))
    # print(deriv_orders)
    # fornberg_diff_mats = get_forrnberg_findiff_mats(s, deriv_orders, acc=acc)
    # fornberg_dr_list = [fornberg_diff_mats[d] @ r for d in deriv_orders]
    FD = FiniteDifferenceOps(
        x=s,
        max_deriv_order=max_deriv_order,
        acc=acc,
        period=b if periodic else None,
        rng=rng,
    )
    # FD.fornberg_findiff_mats = FD.update_fornberg_findiff_mats(periodic=True)
    fornberg_dr_list = [FD.diff(r, deriv_order=d, axis=0) for d in deriv_orders]
    # return fornberg_dr_list

    ndr = r.copy()
    Ndr = [r]
    for n in range(max_deriv_order):
        ndr = np.gradient(ndr, s, edge_order=2, axis=0)
        Ndr.append(ndr)

    fig, axes = plt.subplots(
        1, max_deriv_order - min_deriv_order + 1, figsize=(3 * max_deriv_order, 8)
    )
    title = f"{Ns=}, {jitter=:.2f}, {periodic=}"
    fig.suptitle(title, fontsize=30)
    # print(len(fornberg_dr_list[min_deriv_order:]), len(Ndr[min_deriv_order:]), len(axes))
    d = min_deriv_order
    for rdr, ndr, ax in zip(
        fornberg_dr_list[min_deriv_order:],
        Ndr[min_deriv_order:],
        axes,
    ):
        ax.plot(s, rdr[:, 0], label=f"order {d} fornberg - x")
        # ax.plot(s, ndr[:, 0], linestyle=":", label="numpy - x")
        # ax.plot(s, rdr[:, 1], label="roll - y")
        # ax.plot(s, ndr[:, 1], linestyle=":", label="numpy - y")
        ax.legend()
        d += 1
    plt.show()
    plt.close()
    return FD


def run_diff_fornberg_test():

    FDp = diff_fornberg_test(min_deriv_order=0, max_deriv_order=8, acc=2, periodic=True)

    FDnp = diff_fornberg_test(
        min_deriv_order=0, max_deriv_order=8, acc=2, periodic=False
    )


# %%
# ###########################
# TESTING

# ###########################
# Fornberg finite differences


def get_fornberg_stencil_radius(deriv_order, acc=2, centered=False):
    """
    Minimal radius r (w=2r+1) for target accuracy `acc` on the m-th derivative
    using Fornberg weights. If `centered=True`, even `deriv_order gets` the
    symmetry bonus (+1 order).

    Arguments:
    ----------
    deriv_order (int): order of derivative
    acc (int): order of accuracy of the finite difference approximation
    centered (bool): whether or not to assume symmetric stencil

    Returns:
    -------
        int: radius of stencil
    """
    if centered and (deriv_order % 2 == 0):
        w = deriv_order + acc - 1
    else:
        w = deriv_order + acc
    if w % 2 == 0:
        w += 1
    return (w - 1) // 2


def arclength_centered_stencil_old(
    x,
    i,
    stencil_radius,
    dx_rtol=0.5,
    dx_atol=1e-10,
    x_period=None,
    max_index_radius=12,
):
    """
    Find stencil centered by arclength

    Arguments
    ---------
    x : array_like
        Grid points.
    i : int
        Stencil center index.

    """
    dx = np.zeros(2 * stencil_radius + 1, dtype=float)
    offsets = np.zeros(2 * stencil_radius + 1, dtype=int)

    p = 1
    m = 1
    r = 1  # radius in index space
    jp = i + p
    jm = i - m
    dxm0, dxp0 = 0.0, 0.0
    dxm, dxp = x[jm] - x[i], x[jp] - x[i]
    if dxp < 0 or dxm > 0:
        raise ValueError("Samples must be sorted.")
    dxm *= -1
    while r <= stencil_radius:
        if m > max_index_radius or p > max_index_radius:
            raise ValueError("Could not find stencil.")
        if dxp < dx_rtol * dxm + dx_atol:
            p += 1
            dxp = x[i + p] - x[i]
            if dxp < dxp0:
                raise ValueError("Samples must be sorted.")
            dxp0 = dxp
            continue
        if dxm < dx_rtol * dxp + dx_atol:
            m += 1
            dxm = x[i - m] - x[i]
            if -dxm < dxm0:
                raise ValueError("Samples must be sorted.")
            dxm *= -1
            dxm0 = dxm
            continue
        offsets[stencil_radius - r] = -m
        offsets[stencil_radius + r] = p
        dx[stencil_radius - r] = -dxm
        dx[stencil_radius + r] = dxp

        p += 1
        dxp = x[i + p] - x[i]
        if dxp < dxp0:
            raise ValueError("Samples must be sorted.")
        dxp0 = dxp
        m += 1
        dxm = x[i - m] - x[i]
        if -dxm < dxm0:
            raise ValueError("Samples must be sorted.")
        dxm *= -1
        dxm0 = dxm
        r += 1

    return dx, offsets


def arclength_centered_stencil(x, i, stencil_radius, look_ahead=3):
    """
    Find stencil centered by arclength.

    [
     x[i-m[stencil_radius-1]],
     ...,
     x[i-m[0]],
     x[i],
     x[i+p[0]],
     ...,
     x[i+p[stencil_radius-1]]
    ]

    Offsets m and p are chosen so |x[i-m[s]]-x[i]| ~= |x[i+p[s]]-x[i]|.

    p[0] and m[0] are chosen from 1 <= p[0] <= 1 + look_ahead
    and 1 <= m[0] <= 1 + look_ahead to minimize the difference
    between |x[i-m[0]]-x[i]| and |x[i+p[0]]-x[i]|.

    for 1 <= s <= stencil_radius - 1
        p[s] and m[s] are chosen from

        p[s-1] + 1 <= p[s] <= p[s-1] + 1 + look_ahead

        and

        m[s-1] + 1 <= m[s] <= m[s-1] + 1 + look_ahead

        to minimize the difference between
        |x[i-m[s]]-x[i]| and |x[i+p[s]]-x[i]|

    Arguments
    ---------
    x : array_like
        Grid points.
    i : int
        Stencil center index.
    look_ahead : int
        Width of integer window searched for each offset pair

    """
    dx = np.zeros(2 * stencil_radius + 1, dtype=float)
    pairs = np.zeros((stencil_radius, 2), dtype=int)
    r = 1
    i_p = i + 1
    i_m = i - 1
    while r <= stencil_radius:
        print(f"i={int(i)}, r={int(r)}, i_p={int(i_p)}, i_m={int(i_m)}")
        Ip = np.arange(i_p, i_p + look_ahead + 1)
        Im = np.arange(i_m, i_m - (look_ahead + 1), -1)
        print(f"{Ip=}, {Im=}")
        dXp = x[Ip] - x[i]
        dXm = x[i] - x[Im]
        pair_deltas = np.array(
            [abs(dxp - dxm) for dxp in dXp for dxm in dXm],
            dtype=float,
        )
        sorted_indices = np.argsort(pair_deltas)
        Ipm = np.array(
            [[i_p, i_m] for i_p in Ip for i_m in Im],
            dtype=int,
        )
        i_p, i_m = Ipm[sorted_indices[0]]
        # if ???dx_p, dx_m???
        # dx_p, dx_m = x[i_p] - x[i], x[i] - x[i_m]

        i_p += 1
        i_m -= 1
        pairs[r - 1, :] = Ipm[sorted_indices[0]]
        r += 1
    Indices = np.zeros(2 * stencil_radius + 1, dtype=int)
    Indices[:stencil_radius] = pairs[::-1, 1]
    Indices[stencil_radius] = i
    Indices[stencil_radius + 1 :] = pairs[:, 0]
    offsets = Indices - i
    dx = x[Indices] - x[i]
    return dx, offsets, Indices


def con_test(
    max_deriv_order,  # derivative order to test (1..)
    f,
    N_list,  # iterable of grid sizes, e.g. [50, 100, 200, 400]
    *,
    a=0.0,
    b=1.0,  # domain
    periodic=False,
    radius=None,  # stencil half-width; if None, builder chooses sensible default
    jitter=0.0,  # nonuniformity level (0 = uniform). Perturbs interior points by ~ jitter*mean(dx)
    seed=0,
    norm="inf",  # 'inf' (max), 'l2' (RMS), or 'l1'
    trials=1,  # average over multiple random jitter trials
    # builder=build_diff_operator,
    # applier=apply_diff_operator,
    ax=None,  # optional matplotlib Axes
    label=None,  # curve label
    annotate=True,  # write the slope on the plot
):
    pass


def convergence_test(
    d,  # derivative order to test (1..)
    f,
    df_true,  # callables: y=f(x), y^{(d)}=df_true(x)
    N_list,  # iterable of grid sizes, e.g. [50, 100, 200, 400]
    *,
    a=0.0,
    b=1.0,  # domain
    periodic=False,
    # radius=None,  # stencil half-width; if None, builder chooses sensible default
    acc=2,
    jitter=0.0,  # nonuniformity level (0 = uniform). Perturbs interior points by ~ jitter*mean(dx)
    seed=0,
    norm="inf",  # 'inf' (max), 'l2' (RMS), or 'l1'
    trials=1,  # average over multiple random jitter trials
    # builder=build_diff_operator,
    # applier=apply_diff_operator,
    ax=None,  # optional matplotlib Axes
    label=None,  # curve label
    annotate=True,  # write the slope on the plot
):
    """
    Returns (slope, hs, errs), where slope is the fitted order of accuracy.
    Requires build_diff_operator/apply_diff_operator to be defined.
    """
    # if radius is None:
    #     radius = {1: 1, 2: 2, 3: 2, 4: 3}.get(d, d // 2 + 1)
    rng = np.random.default_rng(seed)
    hs, errs = [], []

    for N in N_list:
        # base grid
        x = np.linspace(a, b, N, endpoint=not periodic)
        # fornberg_dr_list = fornberg_diff(
        #     r,
        #     s,
        #     deriv_order=max_deriv_order,
        #     stencil_rad=stencil_rad,
        #     axis=0,
        # )

        # helper to compute one trial's error
        def one_err(xgrid):
            # op = builder(xgrid, d=d, radius=radius, periodic=periodic)
            y = f(xgrid)
            # num = applier(y, op)
            FD = FiniteDifferenceOps(
                x=xgrid,
                max_deriv_order=d,
                acc=acc,
                period=b if periodic else None,
                rng=rng,
            )
            # Dy = fornberg_diff_OG(
            #     y,
            #     xgrid,
            #     deriv_order=d,
            #     stencil_rad=radius,
            #     axis=0,
            # )
            # num = Dy[d]
            num = FD.diff(y, d)
            true = df_true(xgrid)
            e = np.abs(num - true)
            if norm == "inf":
                return np.max(e)
            elif norm == "l2":
                # RMS with trapezoidal weights (handles nonuniform spacing)
                w = np.empty_like(xgrid)
                w[1:-1] = 0.5 * (xgrid[2:] - xgrid[:-2])
                w[0] = xgrid[1] - xgrid[0]
                w[-1] = xgrid[-1] - xgrid[-2]
                return np.sqrt(np.sum(w * e**2) / (b - a))
            elif norm == "l1":
                w = np.empty_like(xgrid)
                w[1:-1] = 0.5 * (xgrid[2:] - xgrid[:-2])
                w[0] = xgrid[1] - xgrid[0]
                w[-1] = xgrid[-1] - xgrid[-2]
                return np.sum(w * e) / (b - a)
            else:
                raise ValueError("norm must be 'inf', 'l2', or 'l1'.")

        # build (possibly) nonuniform grids and average error over trials
        trial_errs = []
        for t in range(trials):
            if jitter > 0.0:
                xnu = x.copy()
                mean_dx = (b - a) / (N - (0 if periodic else 1))
                # perturb all interior points; keep endpoints fixed for nonperiodic
                if periodic:
                    # allow wrap-around, then sort into [a,b)
                    xnu = xnu + jitter * mean_dx * rng.standard_normal(N)
                    # map to [a,b)
                    L = b - a
                    xnu = a + (xnu - a) % L
                    xnu.sort()
                else:
                    xnu[1:-1] += jitter * mean_dx * rng.standard_normal(N - 2)
                    xnu.sort()
                    xnu[0], xnu[-1] = a, b
            else:
                xnu = x

            trial_errs.append(one_err(xnu))

        err = np.mean(trial_errs)
        errs.append(err)
        # grid size metric: max local spacing (robust for nonuniform)
        hs.append(np.max(np.diff(x if jitter == 0 else xnu)))

    hs = np.array(hs, float)
    errs = np.array(errs, float)

    # fit slope: log(err) = alpha + p*log(h)
    M = np.vstack([np.log(hs), np.ones_like(hs)]).T
    p, alpha = np.linalg.lstsq(M, np.log(errs), rcond=None)[0]

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.loglog(hs, errs, "o-", label=(label or f"d={d}"))
    # reference line through last point with slope p
    href = hs[-1]
    eref = errs[-1]
    hline = np.array([hs.min(), hs.max()])
    eline = eref * (hline / href) ** p
    ax.loglog(hline, eline, "--", linewidth=1)
    if annotate:
        # put a small text near the last point
        ax.text(href, eref, f" slope ≈ {p:.2f}", fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("h (max Δx)")
    ax.set_ylabel(f"‖error‖_{norm}")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    return p, hs, errs


def run_con_tests(acc=2, num_samples=(50, 100, 200, 400, 800)):
    print(2 * "---------------------------\n")
    print("Non-periodic tests\n")
    # Test functions and exact derivatives
    k = 3.0
    f = lambda x: np.sin(k * x)
    df1 = lambda x: k * np.cos(k * x)
    df2 = lambda x: -(k**2) * np.sin(k * x)
    df3 = lambda x: -(k**3) * np.cos(k * x)
    df4 = lambda x: k**4 * np.sin(k * x)

    N_list = list(num_samples)

    p1, hs1, e1 = convergence_test(
        1,
        f,
        df1,
        N_list,
        a=0.0,
        b=1.0,
        periodic=False,
        jitter=0.10,
        trials=3,
        label="f'",
        acc=acc,
    )

    p2, hs2, e2 = convergence_test(
        2,
        f,
        df2,
        N_list,
        a=0.0,
        b=1.0,
        periodic=False,
        jitter=0.10,
        trials=3,
        label="f''",
        acc=acc,
    )

    p3, hs3, e3 = convergence_test(
        3,
        f,
        df3,
        N_list,
        a=0.0,
        b=1.0,
        periodic=False,
        jitter=0.10,
        trials=3,
        label="f'''",
        acc=acc,
    )

    p4, hs4, e4 = convergence_test(
        4,
        f,
        df4,
        N_list,
        a=0.0,
        b=1.0,
        periodic=False,
        jitter=0.10,
        trials=3,
        label="f''''",
        acc=acc,
    )
    plt.show()

    nonperiodic_str = f"Estimated orders (nonperiodic): d1≈{p1:.2f}, d2≈{p2:.2f}, d3≈{p3:.2f}, d4≈{p4:.2f}"
    print(nonperiodic_str + "\n\n")

    print(2 * "---------------------------\n")
    print("Periodic tests\n")
    k = 2.0 * np.pi
    f = lambda x: np.sin(k * x)
    df1 = lambda x: k * np.cos(k * x)
    df2 = lambda x: -(k**2) * np.sin(k * x)
    df3 = lambda x: -(k**3) * np.cos(k * x)
    df4 = lambda x: k**4 * np.sin(k * x)

    # N_list = [50, 100, 200, 400, 800]
    # N_list = [50, 100, 200, 400, 800]

    p1, hs1, e1 = convergence_test(
        1,
        f,
        df1,
        N_list,
        a=0.0,
        b=1.0,
        periodic=True,
        jitter=0.10,
        trials=3,
        label="f'",
        acc=acc,
    )

    p2, hs2, e2 = convergence_test(
        2,
        f,
        df2,
        N_list,
        a=0.0,
        b=1.0,
        periodic=True,
        jitter=0.10,
        trials=3,
        label="f''",
        acc=acc,
    )

    p3, hs3, e3 = convergence_test(
        3,
        f,
        df3,
        N_list,
        a=0.0,
        b=1.0,
        periodic=True,
        jitter=0.10,
        trials=3,
        label="f'''",
        acc=acc,
    )

    p4, hs4, e4 = convergence_test(
        4,
        f,
        df4,
        N_list,
        a=0.0,
        b=1.0,
        periodic=True,
        jitter=0.10,
        trials=3,
        label="f''''",
        acc=acc,
    )
    plt.show()

    periodic_str = f"Estimated orders (periodic): d1≈{p1:.2f}, d2≈{p2:.2f}, d3≈{p3:.2f}, d4≈{p4:.2f}"
    print(periodic_str)

    print(4 * "------------------------\n")

    print(periodic_str + "\n\n")
    print(nonperiodic_str + "\n\n")


# run_con_tests(acc=2, num_samples=(200, 300, 400, 500))


def diff_test():
    # s = np.linspace(0, 1, 333)
    # ds = s[1] - s[0]
    # y = np.sin(3 * np.pi * s)
    # Rdy = [rdiff1(y, ds), rdiff2(y, ds), rdiff3(y, ds), rdiff4(y, ds)]
    # ndy = y.copy()
    # Ndy = []
    # Ddy = []
    # for n in range(4):
    #     ndy = np.gradient(ndy, ds, edge_order=2)
    #     Ndy.append(ndy)
    #     Ddy.append(diff(y, ds, n + 1))
    #
    # fig, axes = plt.subplots(1, 4, figsize=(12, 8))
    #
    # for ddy, rdy, ndy, ax in zip(Ddy, Rdy, Ndy, axes):
    #     ax.plot(s, ddy, label="diff")
    #     ax.plot(s, rdy, label="roll")
    #     # ax.plot(s, ndy, label="numpy")
    #     ax.legend()
    # plt.show()
    # plt.close()

    s = np.linspace(0, 1, 333)
    ds = s[1] - s[0]
    x, y = np.cos(3 * np.pi * s), np.sin(3 * np.pi * s)
    r = np.vstack((x, y)).T
    Rdr = [rdiff1(r, ds), rdiff2(r, ds), rdiff3(r, ds), rdiff4(r, ds)]
    ndr = r.copy()
    Ndr = []
    for n in range(4):
        ndr = np.gradient(ndr, ds, edge_order=2, axis=0)
        Ndr.append(ndr)

    fig, axes = plt.subplots(1, 4, figsize=(12, 8))

    for rdr, ndr, ax in zip(Rdr, Ndr, axes):
        ax.plot(s, rdr[:, 0], label="roll - x")
        ax.plot(s, ndr[:, 0], linestyle=":", label="numpy - x")
        # ax.plot(s, rdr[:, 1], label="roll - y")
        # ax.plot(s, ndr[:, 1], linestyle=":", label="numpy - y")
        ax.legend()
    plt.show()
    plt.close()


def diff_nu_test():

    a, b = 0.0, 1.0
    jitter = 0.1
    Ns = 333
    seed = 0
    rng = np.random.default_rng(seed)
    s = np.linspace(a, b, Ns)
    mean_ds = np.ptp(s) / Ns
    s[1:-1] += jitter * mean_ds * rng.standard_normal(Ns - 2)

    x, y = np.cos(3 * np.pi * s), np.sin(3 * np.pi * s)
    r = np.vstack((x, y)).T
    # Rdr = [rdiff1(r, ds), rdiff2(r, ds), rdiff3(r, ds), rdiff4(r, ds)]
    diff_nu_dr_list = [
        diff_nu(r, s, deriv_order=1, axis=0),
        diff_nu(r, s, deriv_order=2, axis=0),
        diff_nu(r, s, deriv_order=3, axis=0),
        diff_nu(r, s, deriv_order=4, axis=0),
    ]

    ndr = r.copy()
    Ndr = []
    for n in range(4):
        ndr = np.gradient(ndr, s, edge_order=2, axis=0)
        Ndr.append(ndr)

    fig, axes = plt.subplots(1, 4, figsize=(12, 8))

    for rdr, ndr, ax in zip(diff_nu_dr_list, Ndr, axes):
        ax.plot(s, rdr[:, 0], label="diff_nu - x")
        ax.plot(s, ndr[:, 0], linestyle=":", label="numpy - x")
        # ax.plot(s, rdr[:, 1], label="roll - y")
        # ax.plot(s, ndr[:, 1], linestyle=":", label="numpy - y")
        ax.legend()
    plt.show()
    plt.close()
