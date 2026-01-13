import numpy as np


##################
##################
# Misc functions #
##################
##################
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
    if axis != 0:
        y = np.moveaxis(y, axis, 0)

    int_y = 0.5 * (x[1 % x.size] - x[0]) * y[0] + 0.5 * (x[-1] - x[-2 % x.size]) * y[-1]
    for j in range(1, len(y) - 1):
        int_y += 0.5 * (x[j + 1] - x[j - 1]) * y[j]
    if axis != 0:
        int_y = np.moveaxis(int_y, 0, axis)
    return int_y


def trapint_dx(y, dx, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    y = np.asarray(y)
    if axis != 0:
        y = np.moveaxis(y, axis, 0)
    int_y = 0.5 * (dx) * y[0] + 0.5 * (dx) * y[-1]
    for j in range(1, len(y) - 1):
        int_y += 0.5 * (2 * dx) * y[j]
    if axis != 0:
        int_y = np.moveaxis(int_y, 0, axis)
    return int_y


def cumtrapint_dx(y, dx, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    y = np.asarray(y)
    if axis != 0:
        y = np.moveaxis(y, axis, 0)
    int_y = np.zeros_like(y)
    for j in range(1, len(y)):
        int_y[j] = int_y[j - 1] + 0.5 * dx * (y[j] + y[j - 1])
    if axis != 0:
        int_y = np.moveaxis(int_y, 0, axis)
    return int_y


def cumtrapint(y, x, axis=0):
    """
    integrates f over x using trapezoid rule
    """
    y = np.asarray(y)
    if axis != 0:
        y = np.moveaxis(y, axis, 0)
    int_y = np.zeros_like(y)
    for j in range(1, len(y)):
        int_y[j] = int_y[j - 1] + 0.5 * (x[j] - x[j - 1]) * (y[j] + y[j - 1])
    if axis != 0:
        int_y = np.moveaxis(int_y, 0, axis)
    return int_y


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
