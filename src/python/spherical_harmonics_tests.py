import numpy as np
import sympy as sp
from pymathutils.jit_funs import real_Ylm as _jit_real_Ylm
from pymathutils.jit_funs import Ylm as _jit_Ylm
from pymathutils.special import (
    spherical_harmonic_index_n_LM,
    spherical_harmonic_index_lm_N,
    compute_all_series_real_Ylm,
    compute_all_series_Ylm,
    old_compute_all_real_Ylm,
    Ylm,
    real_Ylm,
    compute_all_Ylm,
    compute_all_real_Ylm,
)
from scipy.special import sph_harm_y_all, sph_harm_y, loggamma


def reducedPmm(m, theta):
    if theta == 0:
        if m == 0:
            return 1 / np.sqrt(4 * np.pi)
        else:
            return 0.0
    # s = 1 - 2 * (m & 1)
    # P = np.exp(
    #     m * np.log(np.sin(theta))
    #     - 0.5 * np.log(4 * np.pi)
    #     - m * np.log(2)
    #     + 0.5 * loggamma(2 * m + 2)
    #     - loggamma(m + 1)
    # )
    return (1 - 2 * (m & 1)) * np.exp(
        m * np.log(np.sin(theta))
        - 0.5 * np.log(4 * np.pi)
        - m * np.log(2)
        + 0.5 * loggamma(2 * m + 2)
        - loggamma(m + 1)
    )


def reducedPlm(l, m, theta):
    # m>=0, 0<=theta<=pi/2
    cos_theta = np.cos(theta)
    P_ell_minus_1_m = reducedPmm(m, theta)
    if l == m:
        return P_ell_minus_1_m
    P_ell_m = np.sqrt(2 * m + 3) * cos_theta * P_ell_minus_1_m

    for ell in range(m + 2, l + 1):
        c1 = np.sqrt((2 * ell - 1) * (2 * ell + 1) / ((ell - m) * (ell + m)))
        c0 = np.sqrt(
            (ell - m - 1)
            * (ell + m - 1)
            * (2 * ell + 1)
            / ((ell - m) * (ell + m) * (2 * ell - 3))
        )
        q = c1 * cos_theta * P_ell_m - c0 * P_ell_minus_1_m
        P_ell_minus_1_m = P_ell_m
        P_ell_m = q

    return P_ell_m


def Plm(l, m, theta):
    abs_m = abs(m)
    sigma = 1 - 2 * ((m * (m < 0) + (l - abs_m) * (theta > np.pi / 2)) & 1)
    theta = np.pi / 2 + (theta - np.pi / 2) * (
        int(theta < np.pi / 2) - int(theta > np.pi / 2)
    )
    m = abs_m
    return sigma * reducedPlm(l, m, theta)


def test_spherical_Plm_funs():
    from pymathutils.special import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        compute_all_series_real_Ylm,
        compute_all_series_Ylm,
        old_compute_all_real_Ylm,
        Ylm,
        real_Ylm,
        ReLogRe_ImLogRe_over_pi,
        minus_one_to_int_pow,
        recursive_Ylm,
        recursive_real_Ylm,
        reduced_spherical_Pmm,
        reduced_spherical_Plm,
        spherical_Plm,
    )

    tol = 1e-6
    for l in range(10):
        for m in range(-l, l + 1):
            theta = np.pi * np.random.rand()
            reduced_m = abs(m)
            reduced_theta = min(theta, np.pi - theta)
            reduced_dmm = reducedPmm(reduced_m, reduced_theta) - reduced_spherical_Pmm(
                reduced_m, reduced_theta
            )
            reduced_dlm = reducedPlm(
                l, reduced_m, reduced_theta
            ) - reduced_spherical_Plm(l, reduced_m, reduced_theta)
            dlm = Plm(l, m, theta) - spherical_Plm(l, m, theta)
            if any([abs(val) > tol for val in [reduced_dmm, reduced_dlm, dlm]]):
                print(10 * "-")
                print(f"l={l}, m={m}")
                if abs(reduced_dmm) > tol:
                    print(f"reduced_dmm={reduced_dmm}")
                if abs(reduced_dlm) > tol:
                    print(f"reduced_dlm={reduced_dlm}")
                if abs(dlm) > tol:
                    print(f"dlm={dlm}")

            # if abs(reduced_dmm) > tol:
            #     print(10 * "-")
            #     print(f"l={l}, m={m}")
            #     print(f"reduced_dmm={reduced_dmm}")


def recursiveYlm(l, m, theta, phi):
    return np.exp(1j * m * phi) * Plm(l, m, theta)


def recursiverealYlm(l, m, theta, phi):
    complex_Y = recursiveYlm(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * (-1) ** m * np.imag(complex_Y)
    elif m == 0:
        return np.real(complex_Y)
    else:
        return np.sqrt(2) * (-1) ** m * np.real(complex_Y)


def sympy_precision_scalar_Ylm(l, m, theta_val, phi_val, precision=50):
    theta, phi = sp.symbols("theta phi", real=True)
    ylm_symbolic = sp.Ynm(l, m, theta, phi)
    result = ylm_symbolic.subs(
        {theta: sp.Float(theta_val, precision), phi: sp.Float(phi_val, precision)}
    ).evalf(precision)
    if result.is_real:
        return complex(float(result), 0.0)
    else:
        real_part = float(sp.re(result))
        imag_part = float(sp.im(result))
        return complex(real_part, imag_part)


def sympy_precision_Ylm(l, m, ThetaPhi, precision=50):
    num_points = len(ThetaPhi)
    Y = np.zeros(num_points, dtype=np.complex128)
    for _, (th, ph) in enumerate(ThetaPhi):
        Y[_] = sympy_precision_scalar_Ylm(l, m, th, ph, precision=precision)
    return Y


def jit_Ylm(l, m, ThetaPhi):
    return _jit_Ylm(l, m, *ThetaPhi.T)


def jit_real_Ylm(l, m, ThetaPhi):
    return _jit_real_Ylm(l, m, *ThetaPhi.T)


def sciYlm(l, m, ThetaPhi):
    return sph_harm_y(l, m, *ThetaPhi.T)


def real_sciYlm(l, m, ThetaPhi):
    complex_Y = sph_harm_y(l, abs(m), *ThetaPhi.T)
    if m < 0:
        return np.sqrt(2) * (-1) ** m * np.imag(complex_Y)
    elif m == 0:
        return np.real(complex_Y)
    else:
        return np.sqrt(2) * (-1) ** m * np.real(complex_Y)


def compute_all_sciYlm(l_max, ThetaPhi):
    num_modes = l_max * (l_max + 2) + 1
    num_points = len(ThetaPhi)
    _allY = sph_harm_y_all(l_max, l_max, *ThetaPhi.T)
    allY = np.zeros((num_points, num_modes), dtype=np.complex128)
    for n in range(num_modes):
        l, m = spherical_harmonic_index_lm_N(n)
        allY[:, n] = _allY[l, m, :]
    return allY


def compute_all_real_sciYlm(l_max, ThetaPhi):
    num_modes = l_max * (l_max + 2) + 1
    num_points = len(ThetaPhi)
    _allY = sph_harm_y_all(l_max, l_max, *ThetaPhi.T)
    allY = np.zeros((num_points, num_modes), dtype=np.float64)
    for n in range(num_modes):
        l, m = spherical_harmonic_index_lm_N(n)
        complex_Y = _allY[l, abs(m), :]
        if m < 0:
            allY[:, n] = np.sqrt(2) * (-1) ** m * np.imag(complex_Y)
        elif m == 0:
            allY[:, n] = np.real(complex_Y)
        else:
            allY[:, n] = np.sqrt(2) * (-1) ** m * np.real(complex_Y)
    return allY


def get_problem_theta():
    kinda_small = 1e-6
    small = 1e-9
    really_small = 1e-12
    problem_Theta0 = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    problem_Theta = np.concatenate(
        (
            problem_Theta0,
            problem_Theta0[:-1] + kinda_small,
            problem_Theta0[1:] - kinda_small,
            problem_Theta0[:-1] + small,
            problem_Theta0[1:] - small,
            problem_Theta0[:-1] + really_small,
            problem_Theta0[1:] - really_small,
        )
    )
    problem_Theta = np.sort(problem_Theta)
    return problem_Theta


def test_scalar_Y_vs_Y(Yfun1, Yfun2, tol=1e-8, l_max=100, use_problem_angles=True):
    print(20 * "_")
    print("test_scalar_Y_vs_Y")
    print(f"Yfun1={Yfun1}")
    print(f"Yfun2={Yfun2}")
    print(f"tol={tol}")
    print(f"l_max={l_max}")
    print(f"use_problem_angles={use_problem_angles}\n")

    Theta = np.linspace(0, np.pi, 12)[1:-1]
    Phi = np.linspace(0, 2 * np.pi, 22, endpoint=False)
    if use_problem_angles:
        Theta = np.sort([*get_problem_theta(), *Theta])
    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    num_modes = l_max * (l_max + 2) + 1
    deltainf = np.zeros(num_modes)
    LM = np.zeros((num_modes, 2), dtype=np.int64)

    l_good = -99999
    m_good = -99999
    for n in range(num_modes):
        l, m = spherical_harmonic_index_lm_N(n)
        LM[n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y1 = np.array([Yfun1(l, m, th, ph) for th, ph in ThetaPhi])
        Y2 = np.array([Yfun2(l, m, th, ph) for th, ph in ThetaPhi])
        # Y1, Y2 = [], []
        # for th, ph in ThetaPhi:
        #     # print(f"l={l}, m={m}, theta/pi={th/np.pi}, phi/pi={ph/np.pi}")
        #     Y1.append(Yfun1(l, m, th, ph))
        #     Y2.append(Yfun2(l, m, th, ph))
        # Y1, Y2 = np.array(Y1), np.array(Y2)
        dY = np.abs((Y1 - Y2))
        deltainf[n] = np.linalg.norm(dY, np.inf)
        if deltainf[n] > tol:
            print(f"\nmax error={deltainf[n]}")
            badThetaPhi = ThetaPhi[np.where(dY > tol)] / np.pi
            print("over tolerance at angles (theta, phi):")
            print(badThetaPhi)
            break
        else:
            l_good = l
            m_good = m

    print(
        f"\ngood up to l={l_good}, m={
            m_good} (first {spherical_harmonic_index_n_LM(l_good, m_good)+1} spherical harmonics)"
    )


def test_Y_vs_Y(Yfun1, Yfun2, tol=1e-8, l_max=100, use_problem_angles=True):

    # Yfun1 = Ylm
    # Yfun2 = jit_Ylm
    # tol = 1e-8
    # l_max = 100
    # use_problem_angles = True

    print(20 * "_")
    print("test_Y_vs_Y")
    print(f"Yfun1={Yfun1}")
    print(f"Yfun2={Yfun2}")
    print(f"tol={tol}")
    print(f"l_max={l_max}")
    print(f"use_problem_angles={use_problem_angles}\n")

    Theta = np.linspace(0, np.pi, 12)[1:-1]
    Phi = np.linspace(0, 2 * np.pi, 22, endpoint=False)
    if use_problem_angles:
        Theta = np.sort([*get_problem_theta(), *Theta])
    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    num_modes = l_max * (l_max + 2) + 1
    deltainf = np.zeros(num_modes)
    LM = np.zeros((num_modes, 2), dtype=np.int64)

    l_good = -99999
    m_good = -99999
    for n in range(num_modes):
        l, m = spherical_harmonic_index_lm_N(n)
        LM[n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y1 = Yfun1(l, m, ThetaPhi)
        Y2 = Yfun2(l, m, ThetaPhi)
        dY = np.abs((Y1 - Y2))
        deltainf[n] = np.linalg.norm(dY, np.inf)
        if deltainf[n] > tol:
            print(f"\nmax error={deltainf[n]}")
            badThetaPhi = ThetaPhi[np.where(dY > tol)] / np.pi
            print("over tolerance at angles (theta, phi):")
            print(badThetaPhi)
            break
        else:
            l_good = l
            m_good = m

    print(
        f"\ngood up to l={l_good}, m={
            m_good} (first {spherical_harmonic_index_n_LM(l_good, m_good)+1} spherical harmonics)"
    )


def test_all_Y_vs_Y(
    compute_all_Yfun1, compute_all_Yfun2, tol=1e-8, l_max=100, use_problem_angles=True
):

    # Yfun1 = Ylm
    # Yfun2 = jit_Ylm
    # tol = 1e-8
    # l_max = 100
    # use_problem_angles = True

    print(20 * "_")
    print("test_all_Y_vs_Y")
    print(f"Yfun1={compute_all_Yfun1}")
    print(f"Yfun2={compute_all_Yfun2}")
    print(f"tol={tol}")
    print(f"l_max={l_max}")
    print(f"use_problem_angles={use_problem_angles}\n")

    Theta = np.linspace(0, np.pi, 12)[1:-1]
    Phi = np.linspace(0, 2 * np.pi, 22, endpoint=False)
    if use_problem_angles:
        Theta = np.sort([*get_problem_theta(), *Theta])
    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])

    allY1 = compute_all_Yfun1(l_max, ThetaPhi)
    allY2 = compute_all_Yfun2(l_max, ThetaPhi)
    num_modes = l_max * (l_max + 2) + 1
    deltainf = np.zeros(num_modes)
    LM = np.zeros((num_modes, 2), dtype=np.int64)

    l_good = -99999
    m_good = -99999
    for n in range(num_modes):
        l, m = spherical_harmonic_index_lm_N(n)
        LM[n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y1 = allY1[:, n]
        Y2 = allY2[:, n]
        dY = np.abs((Y1 - Y2))
        deltainf[n] = np.linalg.norm(dY, np.inf)
        if deltainf[n] > tol:
            print(f"\nmax error={deltainf[n]}")
            badThetaPhi = ThetaPhi[np.where(dY > tol)] / np.pi
            print("over tolerance at angles (theta, phi):")
            print(badThetaPhi)
            break
        else:
            l_good = l
            m_good = m

    print(
        f"\ngood up to l={l_good}, m={
            m_good} (first {spherical_harmonic_index_n_LM(l_good, m_good)+1} spherical harmonics)"
    )


def time_all_Y_vs_Y(
    compute_all_Yfun1, compute_all_Yfun2, l_max=100, use_problem_angles=True
):

    # Yfun1 = Ylm
    # Yfun2 = jit_Ylm
    # tol = 1e-8
    # l_max = 100
    # use_problem_angles = True

    print(20 * "_")
    print("time_all_Y_vs_Y")
    print(f"Yfun1={compute_all_Yfun1}")
    print(f"Yfun2={compute_all_Yfun2}")
    print(f"l_max={l_max}")
    print(f"use_problem_angles={use_problem_angles}\n")

    Theta = np.linspace(0, np.pi, 12)[1:-1]
    Phi = np.linspace(0, 2 * np.pi, 22, endpoint=False)
    if use_problem_angles:
        Theta = np.sort([*get_problem_theta(), *Theta])
    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])

    print("%timeit compute_all_Yfun1(l_max, ThetaPhi)")
    # %timeit compute_all_Yfun1(l_max, ThetaPhi)
    print("%timeit compute_all_Yfun2(l_max, ThetaPhi)")
    # %timeit compute_all_Yfun2(l_max, ThetaPhi)


def run_precision_tests():
    tol = 1e-12
    # test_scalar_Y_vs_Y(Ylm, sph_harm_y, tol=tol, l_max=100, use_problem_angles=True)

    # test_Y_vs_Y(Ylm, sympy_precision_Ylm, tol=tol, l_max=20, use_problem_angles=True)
    # test_Y_vs_Y(sciYlm, sympy_precision_Ylm, tol=tol, l_max=20, use_problem_angles=True)

    # test_Y_vs_Y(Ylm, jit_Ylm, tol=tol, l_max=100, use_problem_angles=True)
    # test_Y_vs_Y(real_Ylm, jit_real_Ylm, tol=tol, l_max=100, use_problem_angles=True)
    test_Y_vs_Y(Ylm, sciYlm, tol=tol, l_max=200, use_problem_angles=True)
    test_Y_vs_Y(real_Ylm, real_sciYlm, tol=tol, l_max=200, use_problem_angles=True)
    test_all_Y_vs_Y(
        compute_all_Ylm,
        compute_all_sciYlm,
        tol=tol,
        l_max=200,
        use_problem_angles=True,
    )
    test_all_Y_vs_Y(
        compute_all_real_Ylm,
        compute_all_real_sciYlm,
        tol=tol,
        l_max=200,
        use_problem_angles=True,
    )


run_precision_tests()
