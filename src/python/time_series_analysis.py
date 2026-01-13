import struct, numpy as np
from scipy.interpolate import make_splrep
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# from src_python.pybrane.combinatorics.permutations import inverse as perm_inv


def crosscorr(
    X,
    W,
    D,
    start=0,
    stop=None,
    step=1,
):
    r"""
    Compute slice of
    $$
    \tilde{X}_{j...} = \sum_{m} w_{m}X_{(j+d_{m})_{\mod N}...j}
    $$
    where $w\in\mathbb{R}^M$ and $X \in \mathbb{R}^{N\times...}$.

    Args
    ----
        X : ndarray
        W : ndarray
        D : ndarray
    Returns
    -------
        ndarray : The correlated signals.
    """

    if len(W) != len(D):
        raise ValueError(r"#W != #D")

    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    D = np.asarray(D, dtype=int)
    num_X = len(X)

    if stop is None:
        stop = num_X

    return np.array([W @ X[(j + D) % num_X] for j in range(start, stop, step)])


def moving_average(
    samples,
    window_size,
    window_type="backward",
    periodic=False,
    axis=0,
    verbose=False,
):
    """
    Compute the moving average of a 1D array.

    Args
    ----
        samples (ndarray) : The input samples.
        window_size (int) : The size of the moving window.

    Returns
    -------
        np.ndarray: The moving average values.
    """
    X = np.asarray(samples)

    if window_type == "backward":
        W = np.ones(window_size, dtype=float) / window_size
        D = np.array(range(-window_size + 1, 1))
        if periodic:
            return crosscorr(X, W, D)
        data_out = np.empty_like(X)
        data_out[window_size - 1 :] = crosscorr(X, W, D, start=window_size - 1)
        for n in range(0, window_size - 1):
            k0 = max(0, n - window_size + 1)
            dk = n - k0 + 1
            data_out[n] = sum(X[k0 : n + 1]) / dk
        return data_out
    elif window_type == "centered":
        if window_size % 2 == 0:
            window_size += 1
        W = np.ones(window_size, dtype=float) / window_size
        h = (window_size - 1) // 2
        D = np.array(range(-h, h + 1))
        if periodic:
            return crosscorr(X, W, D)
        data_out = np.empty_like(X)
        N = len(X)
        data_out[h : N - h] = crosscorr(X, W, D, start=h, stop=N - h)

        for n in range(0, h):
            hh = n
            k0 = 0
            k1 = n + hh
            K = np.array([k for k in range(k0, k1 + 1)], dtype=int)
            data_out[n] = sum(samples[K]) / len(K)
        for n in range(N - 1 - h, N):
            hh = N - 1 - n
            k0 = n - hh
            k1 = N - 1
            K = np.array([k for k in range(k0, k1 + 1)], dtype=int)
            data_out[n] = sum(samples[K]) / len(K)
        return data_out
    elif window_type == "forward":
        W = np.ones(window_size, dtype=float) / window_size
        D = np.array(range(0, window_size))
        if periodic:
            return crosscorr(X, W, D)
        data_out = np.empty_like(X)
        N = len(X)
        data_out[: N - (window_size - 1)] = crosscorr(X, W, D, stop=N - (window_size - 1))
        for n in range(N - (window_size - 1), N):
            data_out[n] = sum(X[n:N]) / (N - n)
        return data_out

    else:
        raise NotImplementedError("window_type must be one of: 'backward', 'centered', 'forward'")


def arclength2d(x, y):
    """
    Cumulative arclenth for a 2d curve.

    Args
    ----
        x (ndarrray) : x-coordinate samples
        y (ndarrray) : y-coordinate samples
    Returns
    -------
        s (ndarray) : arclength samples
        ds (ndarray) : edge lengths
    """
    ds = np.zeros_like(x)
    ds[1:] = np.linalg.norm(np.diff(np.array([x, y]), axis=1), axis=0)
    s = np.cumsum(ds)
    return s, ds


def read_time_series(filepath, verbose=False):
    """
    Read output files (.dat) from rigid spindle sims.

    Args
    ----
        filepath (str) : path to .dat file
    Returns
    -------
        ndarray : time series

    """

    try:
        with open(filepath, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            rows = struct.unpack("<q", f.read(8))[0]
            cols = struct.unpack("<q", f.read(8))[0]
            data = np.fromfile(f, dtype=np.float64)
        # reshape into (n, rows, cols) because we wrote row-major blocks
        data = data.reshape((n, rows, cols))
        # data[i] is the Samples2d for frame i
        if verbose:
            print(f"Opened array-valued time series {filepath}")
        return data

    except ValueError:

        with open(filepath, "rb") as f:
            # Read the size of the vector (size_t is typically 8 bytes on 64-bit systems)
            size = np.fromfile(f, dtype=np.uint64, count=1)[0]
            # Read the data
            data = np.fromfile(f, dtype=np.float64, count=size)
        if verbose:
            print(f"Opened scalar-valued time series {filepath}")
        return data


def arclength_splines2d(x, y, smooth_factor=1e-3):
    """
    Arclength parameterized splines for a 2d curve.

    Args
    ----
        x (ndarrray) : x-coordinate samples
        y (ndarrray) : y-coordinate samples
        smooth_factor (optional, float) :
    """
    ds = np.zeros_like(x)
    ds[1:] = np.linalg.norm(np.diff(np.array([x, y]), axis=1), axis=0)
    var_ds = np.var(ds)
    smooth_param = smooth_factor * len(ds) * var_ds
    s = np.cumsum(ds)
    spline_x = make_splrep(s, x, k=3, s=smooth_param)
    spline_y = make_splrep(s, y, k=3, s=smooth_param)
    return {
        "x": spline_x,
        "y": spline_y,
        "s": s,
        "ds": ds,
        "var_ds": var_ds,
        "smooth_param": smooth_param,
        "smooth_factor": smooth_factor,
    }


def smooth_samples(
    samples,
    window_factor=0.1,
    smooth_type="moving average",
    periodic=False,
    num_passes=1,
):
    if smooth_type == "savgol":
        win = int(window_factor * len(samples) / num_passes)
        win = min(win, 2)
        polyorder = min(win - 1, 3)
        for _ in range(num_passes):
            samples = savgol_filter(samples, window_length=win, polyorder=polyorder)
        return samples
    if smooth_type == "moving average":
        win = int(window_factor * len(samples) / num_passes)
        for _ in range(num_passes):
            samples = moving_average(
                samples,
                win,
                window_type="centered",
                periodic=periodic,
            )
        return samples
    raise ValueError


def barycentric_collapse2d(x, y, num_keep=250, periodic=False):
    """
    Compute barycenters of `num_keep` bins, evenly spaced by arclength.
    Assumes points are sorted by arclength, but not evenly spaced.
    """
    num_x, num_y = len(x), len(y)
    if any([num_x != num_y, num_x < num_keep]):
        raise ValueError(f"{len(x)=} must equal {len(y)=} and be greater than {num_keep=}.")
    if type(num_keep) != int:
        raise ValueError(f"{num_keep=} must be `int` type.")

    if periodic:
        s, ds_all = arclength2d(x, y)
        smin, smax = s[0], s[-1]
        period = s[-1] + np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2)

        X, Y, S = np.zeros_like(x, shape=(3, num_keep))

        ds_bin = period / num_keep
        nb = 0
        s_start, s_stop = smin, smin + ds_bin
        max_expands = 10 * (num_keep + 1)
        expands = 0
        while nb < num_keep:
            bin_mask = np.logical_or(
                np.logical_and(s_start <= s, s < s_stop),
                np.logical_and(s_start - period <= s, s < s_stop - period),
            )

            if not np.any(bin_mask):
                s_stop += ds_bin
                expands += 1
                if expands > max_expands:
                    raise RuntimeError("Unable to fill bins. ")
                continue

            X[nb] = np.mean(x[bin_mask])
            Y[nb] = np.mean(y[bin_mask])
            S[nb] = np.mean(s[bin_mask])
            nb += 1
            s_start = s_stop
            s_stop += ds_bin
            expands = 0

        # plt.plot(S, X, marker=".", label="X(S) - periodic")
        # plt.plot(S, Y, marker=".", label="Y(S) - periodic")
        # plt.legend()
        # plt.show()
        # plt.close()
        return (X, Y, S)
    else:
        s, ds_all = arclength2d(x, y)
        smin, smax = s[0], s[-1]

        X, Y, S = np.zeros_like(x, shape=(3, num_keep))

        ds_bin = (smax - smin) / num_keep
        nb = 0
        s_start, s_stop = smin, smin + ds_bin
        max_expands = 10 * (num_keep + 1)
        expands = 0
        while nb < num_keep:
            # print(f"{s_stop=}")
            bin_mask = np.logical_and(s_start <= s, s < s_stop)

            if not np.any(bin_mask):
                s_stop += ds_bin
                expands += 1
                if expands > max_expands:
                    raise RuntimeError("Unable to fill bins. ")
                continue

            X[nb] = np.mean(x[bin_mask])
            Y[nb] = np.mean(y[bin_mask])
            S[nb] = np.mean(s[bin_mask])
            nb += 1
            s_start = s_stop
            s_stop += ds_bin
            expands = 0

        # plt.plot(S, X, marker=".", label="X(S)")
        # plt.plot(S, Y, marker=".", label="Y(S)")
        # plt.legend()
        # plt.show()
        # plt.close()
        return (X, Y, S)


def interp_zr_coords_V(
    zr_coords_V,
    num_interp=50,
    num_plot=300,
    smooth_factor=0.0004,
    window_factor=0.15,
    smooth_type="savgol",
):
    z_cloud0, r_cloud0 = zr_coords_V.T
    argsort_z = np.argsort(z_cloud0)
    z_cloud, r_cloud = z_cloud0[argsort_z], r_cloud0[argsort_z]
    z_cloud = np.array([*z_cloud, *z_cloud[-2::-1]])
    r_cloud = np.array([*r_cloud, *(-r_cloud[-2::-1])])
    # Thin point cloud, get splines
    z_thin, r_thin, s_thin = barycentric_collapse2d(z_cloud, r_cloud, num_interp, periodic=True)

    # z_thin = np.array([*z_thin, *z_thin[-2::-1]])
    # r_thin = np.array([*r_thin, *(-r_thin[-2::-1])])

    splines = arclength_splines2d(z_thin, r_thin, smooth_factor=smooth_factor)

    s = np.linspace(np.min(splines["s"]), np.max(splines["s"]), num_plot)
    z_spline = splines["x"](s)
    r_spline = splines["y"](s)

    # splines = arclength_splines2d(z_spline, r_spline, smooth_factor=smooth_factor)
    # s = np.linspace(np.min(splines["s"]), np.max(splines["s"]), num_plot)
    # z_spline = splines["x"](s)
    # r_spline = splines["y"](s)

    z_smooth = smooth_samples(
        z_spline,
        window_factor=window_factor,
        # smooth_type="moving average",
        smooth_type=smooth_type,
        periodic=True,
        num_passes=1,
    )
    r_smooth = smooth_samples(
        r_spline,
        window_factor=window_factor,
        # smooth_type="moving average",
        smooth_type=smooth_type,
        periodic=True,
        num_passes=1,
    )
    z_smooth = np.array([*z_smooth, z_smooth[0]])
    r_smooth = np.array([*r_smooth, r_smooth[0]])


def plot_zr_coords_V(
    data_dir="output/rigid_spindle_test/raw_data",
    view_coords=[5, 1],
    num_interp=100,
    num_plot=300,
    smooth_factor=0.0004,
    show_point_cloud=False,
    Nt_skip=10,
    marker=None,
    lw=1,
    markersize=2.5,
    window_factor=0.15,
    # periodic=True
    smooth_type="savgol",
):
    """
    Args
    ----
    num_interp : int
        Number of samples to use for computing splines.
    """

    filepath = f"{data_dir}/envelope_zr_coords.dat"
    data = read_time_series(filepath)
    # num_pts = len(data[0])
    print(f"{len(data[0])=}")

    for _ in range(0, len(data), Nt_skip):
        # Sort point cloud by `z`
        zr_coords_V = data[_]
        z_cloud0, r_cloud0 = zr_coords_V.T
        argsort_z = np.argsort(z_cloud0)
        z_cloud, r_cloud = z_cloud0[argsort_z], r_cloud0[argsort_z]
        z_cloud = np.array([*z_cloud, *z_cloud[-2::-1]])
        r_cloud = np.array([*r_cloud, *(-r_cloud[-2::-1])])
        # Thin point cloud, get splines
        z_thin, r_thin, s_thin = barycentric_collapse2d(z_cloud, r_cloud, num_interp, periodic=True)

        # z_thin = np.array([*z_thin, *z_thin[-2::-1]])
        # r_thin = np.array([*r_thin, *(-r_thin[-2::-1])])

        splines = arclength_splines2d(z_thin, r_thin, smooth_factor=smooth_factor)
        s = np.linspace(np.min(splines["s"]), np.max(splines["s"]), num_plot)
        z_spline = splines["x"](s)
        r_spline = splines["y"](s)

        # splines = arclength_splines2d(z_spline, r_spline, smooth_factor=smooth_factor)
        # s = np.linspace(np.min(splines["s"]), np.max(splines["s"]), num_plot)
        # z_spline = splines["x"](s)
        # r_spline = splines["y"](s)

        z_smooth = smooth_samples(
            z_spline,
            window_factor=window_factor,
            smooth_type=smooth_type,
            # smooth_type="savgol",
            periodic=True,
            num_passes=1,
        )
        r_smooth = smooth_samples(
            r_spline,
            window_factor=window_factor,
            smooth_type=smooth_type,
            # smooth_type="savgol",
            periodic=True,
            num_passes=1,
        )
        z_smooth = np.array([*z_smooth, z_smooth[0]])
        r_smooth = np.array([*r_smooth, r_smooth[0]])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 12))
        if show_point_cloud:
            ax.plot(z_thin, r_thin, ".", markersize=markersize, label="cloud")

        # ax.plot(z_spline, r_spline, marker=marker, markersize=markersize, lw=lw, label="spline")
        ax.plot(z_smooth, r_smooth, marker=marker, markersize=markersize, lw=lw, label="smooth")

        ax.set_aspect("equal", adjustable="box")

        ax.plot(
            [-view_coords[0], view_coords[0], view_coords[0], -view_coords[0]],
            [-view_coords[1], -view_coords[1], view_coords[1], view_coords[1]],
            lw=0,
        )
        plt.tight_layout()
        # plt.legend()
        plt.show()


def plot_zr_coords_V_with_stuff(
    data_dir="output/rigid_spindle_test/raw_data",
    view_coords=[5, 1],
    num_interp=100,
    num_plot=300,
    smooth_factor=0.0004,
    show_point_cloud=False,
    Nt_skip=10,
    marker=None,
    lw=1,
    markersize=2.5,
    window_factor=0.15,
    # periodic=True
    smooth_type="savgol",
    nt_window=50,
):
    """
    Args
    ----
    num_interp : int
        Number of samples to use for computing splines.
    """

    filepath = f"{data_dir}/envelope_zr_coords.dat"
    data = read_time_series(filepath)
    # num_pts = len(data[0])
    print(f"{len(data[0])=}")

    data_names = [
        "t",
        "mt_bundle_length",
        "mt_bundle_length_dot",
        "mt_overlap_length",
        "mt_overlap_length_dot",
        "extensile_force",
        "envelope_midpoint_radius",
        "compressive_force",
        "envelope_compressive_force",
    ]
    filepath_dict = {_: f"{data_dir}/{_}.dat" for _ in data_names}
    data_dict = {k: read_time_series(v) for k, v in filepath_dict.items()}
    data_dict0 = data_dict.copy()
    data_dict = {
        k: moving_average(
            v,
            nt_window,
            window_type="centered",
        )
        for k, v in data_dict.items()
    }
    nt_cut = int(nt_window / 2)

    t = data_dict["t"]
    overlap_length = data_dict["mt_overlap_length"]
    compressive_force = data_dict["compressive_force"]

    for nt in range(0, len(data), Nt_skip):
        # Sort point cloud by `z`
        zr_coords_V = data[nt]
        z_cloud0, r_cloud0 = zr_coords_V.T
        argsort_z = np.argsort(z_cloud0)
        z_cloud, r_cloud = z_cloud0[argsort_z], r_cloud0[argsort_z]
        z_cloud = np.array([*z_cloud, *z_cloud[-2::-1]])
        r_cloud = np.array([*r_cloud, *(-r_cloud[-2::-1])])
        # Thin point cloud, get splines
        z_thin, r_thin, s_thin = barycentric_collapse2d(z_cloud, r_cloud, num_interp, periodic=True)

        # z_thin = np.array([*z_thin, *z_thin[-2::-1]])
        # r_thin = np.array([*r_thin, *(-r_thin[-2::-1])])

        splines = arclength_splines2d(z_thin, r_thin, smooth_factor=smooth_factor)
        s = np.linspace(np.min(splines["s"]), np.max(splines["s"]), num_plot)
        z_spline = splines["x"](s)
        r_spline = splines["y"](s)

        # splines = arclength_splines2d(z_spline, r_spline, smooth_factor=smooth_factor)
        # s = np.linspace(np.min(splines["s"]), np.max(splines["s"]), num_plot)
        # z_spline = splines["x"](s)
        # r_spline = splines["y"](s)

        z_smooth = smooth_samples(
            z_spline,
            window_factor=window_factor,
            smooth_type=smooth_type,
            # smooth_type="savgol",
            periodic=True,
            num_passes=1,
        )
        r_smooth = smooth_samples(
            r_spline,
            window_factor=window_factor,
            smooth_type=smooth_type,
            # smooth_type="savgol",
            periodic=True,
            num_passes=1,
        )
        z_smooth = np.array([*z_smooth, z_smooth[0]])
        r_smooth = np.array([*r_smooth, r_smooth[0]])

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
        ax = axes[0]
        if show_point_cloud:
            ax.plot(z_thin, r_thin, ".", markersize=markersize, label="cloud")

        # ax.plot(z_spline, r_spline, marker=marker, markersize=markersize, lw=lw, label="spline")
        ax.plot(z_smooth, r_smooth, marker=marker, markersize=markersize, lw=lw, label="smooth")

        ax.set_aspect("equal", adjustable="box")

        ax.plot(
            [-view_coords[0], view_coords[0], view_coords[0], -view_coords[0]],
            [-view_coords[1], -view_coords[1], view_coords[1], view_coords[1]],
            lw=0,
        )

        if nt > nt_cut:
            ax = axes[1]
            ax.plot(t[nt_cut:nt], overlap_length[nt_cut:nt], label="Overlap length")
            ax.set_ylabel("Overlap Length")
            ax.legend()
            ax = axes[2]
            ax.plot(t[nt_cut:nt], compressive_force[nt_cut:nt], label="Compressive force")
            ax.set_ylabel("Force")
            ax.legend()

        plt.tight_layout()
        # plt.legend()
        plt.show()


def barycentric_collapse2d1(x, y, num_keep=250, periodic=False):
    """
    Compute barycenters of `num_keep` bins, evenly spaced by arclength.
    Assumes points are sorted by arclength, but not evenly spaced.
    """
    num_x, num_y = len(x), len(y)
    if any([num_x != num_y, num_x < num_keep]):
        raise ValueError(f"{len(x)=} must equal {len(y)=} and be greater than {num_keep=}.")
    if type(num_keep) != int:
        raise ValueError(f"{num_keep=} must be `int` type.")

    if periodic:
        s, ds_all = arclength2d(x, y)
        smin, smax = s[0], s[-1]
        period = s[-1] + np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2)
        # remove points until remaining number is a multiple of num_keep
        Ns = num_keep * (len(s) // num_keep)  # floor(Ns0/Nkeep)*Nkeep
        # while len(s) != Ns:
        #     i = np.random.randint(0, len(s))
        #     s = np.delete(s, i)
        #     x = np.delete(x, i)
        #     y = np.delete(y, i)
        X, Y, S = np.zeros_like(x, shape=(3, num_keep))

        ds_bin = period / num_keep
        nb = 0
        s_start, s_stop = smin, smin + ds_bin
        while nb < num_keep:
            bin_mask = np.logical_or(
                np.logical_and(s_start <= s, s < s_stop),
                np.logical_and(s_start - period <= s, s < s_stop - period),
            )

            if not np.any(bin_mask):
                s_stop += ds_bin
                continue

            X[nb] = np.mean(x[bin_mask])
            Y[nb] = np.mean(y[bin_mask])
            S[nb] = np.mean(s[bin_mask])
            nb += 1
            s_start = s_stop
            s_stop += ds_bin

        # plt.plot(S, X, marker=".", label="X(S) - periodic")
        # plt.plot(S, Y, marker=".", label="Y(S) - periodic")
        # plt.legend()
        # plt.show()
        # plt.close()
        return (X, Y, S)
    else:
        s, ds_all = arclength2d(x, y)
        smin, smax = s[0], s[-1]
        # Ns0 = len(s)
        # dN = len(s) // num_keep  # floor(Ns0/Nkeep)
        Ns = num_keep * (len(s) // num_keep)  # floor(Ns0/Nkeep)*Nkeep
        while len(s) != Ns:
            i = np.random.randint(1, len(s) - 1)  # never delete boundary points
            s = np.delete(s, i)
            x = np.delete(x, i)
            y = np.delete(y, i)
        X, Y, S = np.zeros_like(x, shape=(3, num_keep))

        ds_bin = (smax - smin) / num_keep
        nb = 0
        s_start, s_stop = smin, smin + ds_bin
        while nb < num_keep:
            # print(f"{s_stop=}")
            bin_mask = np.logical_and(s_start <= s, s < s_stop)

            if not np.any(bin_mask):
                s_stop += ds_bin
                continue

            X[nb] = np.mean(x[bin_mask])
            Y[nb] = np.mean(y[bin_mask])
            S[nb] = np.mean(s[bin_mask])
            nb += 1
            s_start = s_stop
            s_stop += ds_bin

        # plt.plot(S, X, marker=".", label="X(S)")
        # plt.plot(S, Y, marker=".", label="Y(S)")
        # plt.legend()
        # plt.show()
        # plt.close()
        return (X, Y, S)


def barycentric_collapse2d0(x, y, num_keep=250, periodic=False):

    I = np.argsort(x)
    x, y = x[I], y[I]
    s, ds0 = arclength2d(x, y)

    # plt.plot(s, s, ".", label="s")
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # plt.plot(s, x, label="x(s)")
    # plt.plot(s, y, label="y(s)")
    # plt.legend()
    # plt.show()
    # plt.close()

    # N = len(x)
    Ns0 = len(s)
    dN = len(s) // num_keep
    Ns = int(num_keep * dN)
    while len(s) != Ns:
        i = np.random.randint(len(s))
        s = np.delete(s, i)
        x = np.delete(x, i)
        y = np.delete(y, i)

    # print(len)
    N = num_keep
    xx, yy = -np.ones((2, N))
    ss = -np.ones(N)
    ss[0] = s[0]
    xx[0] = x[0]
    yy[0] = y[0]

    ds = np.ptp(s) / N
    ddds = (s[-1] - s[0]) / N
    assert ds == ddds

    smin, smax = s[0], s[-1]
    assert smin == np.min(s) and smax == np.max(s)
    s0, s1 = s[0], s[0]
    n = 1
    while n < N and s1 < smax:
        # while s0 < smax:
        s0 = s1
        s1 += ds
        I = np.logical_and(s > s0, s <= s1)
        X, Y, S = x[I], y[I], s[I]
        if len(S) == 0:
            print(f"no points in interval {n=}, {s0=}, {s1=}")
            continue

        xx[n] = np.mean(X)
        yy[n] = np.mean(Y)
        ss[n] = np.mean(S)

        # print(f"{ss[n]=}")

        # if ss[n - 1] > ss[n]:
        #     if n - 1 < 0:
        #         pass
        #     else:
        #         ss[n - 1] = (ss[n - 2] + ss[n]) / 2

        n += 1

    ss[-1] = s[-1]
    # print(f"{s[-1]=}")
    # print(f"{ss[-1]=}")

    # Iz = np.argsort(zz)
    # zz, rr = zz[Iz], rr[Iz]

    # plt.plot(ss, ss, ".", label="ss")
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # plt.plot(ss, xx, ".", label="xx(ss)")
    # plt.plot(ss, yy, ".", label="yy(ss)")
    # plt.legend()
    # plt.show()
    # plt.close()

    num_undef = len([_ for _ in ss if _ == -1])
    undef = [i for i, si in enumerate(ss) if si == -1]
    # print(f"{undef=}")

    # num_undef = len(np.where(ss == -1)[0])
    # print(f"{np.where(ss == -1)[0]=}")

    assert ss[0] != ss[1]
    assert ss[-1] != ss[-2]

    assert s[0] == ss[0]
    assert s[-1] == ss[-1]

    return (xx, yy, ss)


def moving_average0(
    data,
    window_size,
    window_type="backward",
    periodic=False,
    orig_weight=0.5,
    verbose=False,
):
    """
    Compute the moving average of a 1D array.

    Parameters:
        data (list or np.ndarray): The input data.
        window_size (int): The size of the moving window.

    Returns:
        ndarray: The moving average values.
    """
    if window_type == "backward":
        data_out = np.zeros_like(data)
        for n in range(0, window_size - 1):
            k0 = max(0, n - window_size + 1)
            dk = n - k0 + 1
            data_out[n] = sum(data[k0 : n + 1]) / dk
        data_out[window_size - 1 :] = np.convolve(data, np.ones(window_size) / window_size, mode="valid")

        return data_out
    if window_type == "centered" and periodic:

        w = window_size
        if w % 2 == 0:
            w += 1
        h = (w - 1) // 2
        weights = np.array([1.0 - orig_weight if _ != h else orig_weight for _ in range(w)])
        weights /= sum(weights)

        # print(f"{w=}, {h=}")
        data_out = np.zeros_like(data)
        N = len(data)
        for n in range(0, N):
            K = np.array([k % N for k in range(n - h, n + h + 1)], dtype=int)
            # print(f"{K=}")
            data_out[n] = weights @ data[K]
        return data_out
    if window_type == "centered" and not periodic:
        w = window_size
        if w % 2 == 0:
            w += 1
        h = (w - 1) // 2
        # print(f"{w=}, {h=}")
        data_out = np.zeros_like(data)
        N = len(data)
        for n in range(h, N - h):
            K = np.array([k % N for k in range(n - h, n + h + 1)], dtype=int)
            # print(f"{K=}")
            data_out[n] = sum(data[K]) / len(K)
        for n in range(0, h):
            hh = n
            k0 = 0
            k1 = n + hh
            K = np.array([k for k in range(k0, k1 + 1)], dtype=int)
            # W =
            # print(f"{K=}")
            data_out[n] = sum(data[K]) / len(K)
        for n in range(N - 1 - h, N):
            hh = N - 1 - n
            k0 = n - hh
            k1 = N - 1
            K = np.array([k for k in range(k0, k1 + 1)], dtype=int)
            # W =
            # print(f"{K=}")
            data_out[n] = sum(data[K]) / len(K)

        return data_out


def test_moving_average():
    num_pts = 1600
    window_size = 101

    x = np.linspace(0, 1, num_pts, endpoint=False)
    y = np.sin(4.3 * np.pi * x)

    avy_back = moving_average(
        y,
        window_size=window_size,
        window_type="backward",
        verbose=True,
    )
    avy_cent = moving_average(
        y,
        window_size=window_size,
        window_type="centered",
    )
    avy_forw = moving_average(
        y,
        window_size=window_size,
        window_type="forward",
    )

    avy_per_back = moving_average(
        y,
        window_size=window_size,
        window_type="backward",
        periodic=True,
    )
    avy_per_cent = moving_average(
        y,
        window_size=window_size,
        window_type="centered",
        periodic=True,
    )
    avy_per_forw = moving_average(
        y,
        window_size=window_size,
        window_type="forward",
        periodic=True,
    )

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    fig.suptitle(f"{num_pts=}, {window_size=}")

    axes[0, 0].plot(
        x,
        y,
        # marker=".",
        label="y",
    )
    axes[0, 0].plot(
        x,
        avy_back,
        # marker=".",
        label="avy",
    )
    axes[0, 0].set_title("backward")
    axes[0, 0].legend()
    axes[0, 1].plot(
        x,
        y,
        # marker=".",
        label="y",
    )
    axes[0, 1].plot(
        x,
        avy_cent,
        # marker=".",
        label="avy",
    )
    axes[0, 1].set_title("centered")
    axes[0, 1].legend()
    axes[0, 2].plot(
        x,
        y,
        # marker=".",
        label="y",
    )
    axes[0, 2].plot(
        x,
        avy_forw,
        # marker=".",
        label="avy",
    )
    axes[0, 2].set_title("forward")
    axes[0, 2].legend()
    axes[1, 0].plot(
        x,
        y,
        # marker=".",
        label="y",
    )
    axes[1, 0].plot(
        x,
        avy_per_back,
        # marker=".",
        label="avy",
    )
    axes[1, 0].set_title("backward, periodic")
    axes[1, 0].legend()
    axes[1, 1].plot(
        x,
        y,
        # marker=".",
        label="y",
    )
    axes[1, 1].plot(
        x,
        avy_per_cent,
        # marker=".",
        label="avy",
    )
    axes[1, 1].set_title("centered, periodic")
    axes[1, 1].legend()
    axes[1, 2].plot(
        x,
        y,
        # marker=".",
        label="y",
    )
    axes[1, 2].plot(
        x,
        avy_per_forw,
        # marker=".",
        label="avy",
    )
    axes[1, 2].set_title("forward, periodic")
    axes[1, 2].legend()

    plt.show()
    plt.close


# test_moving_average()
