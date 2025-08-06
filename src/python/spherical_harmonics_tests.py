import numpy as np
import sympy as sp
from mathutils.jit_funs import real_Ylm as _jit_real_Ylm
from mathutils.jit_funs import Ylm as _jit_Ylm
from mathutils.special import (
    spherical_harmonic_index_n_LM,
    spherical_harmonic_index_lm_N,
    compute_all_real_Ylm,
    compute_all_Ylm,
    old_compute_all_real_Ylm,
    Ylm,
    real_Ylm,
    ReLogRe_ImLogRe_over_pi,
    minus_one_to_int_pow,
)
from scipy.special import sph_harm_y_all, sph_harm_y


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
        problem_Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        Theta = np.array([*problem_Theta, *Theta])
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
    print("test_Y_vs_Y")
    print(f"Yfun1={compute_all_Yfun1}")
    print(f"Yfun2={compute_all_Yfun2}")
    print(f"tol={tol}")
    print(f"l_max={l_max}")
    print(f"use_problem_angles={use_problem_angles}\n")

    Theta = np.linspace(0, np.pi, 12)[1:-1]
    Phi = np.linspace(0, 2 * np.pi, 22, endpoint=False)
    if use_problem_angles:
        problem_Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        Theta = np.array([*problem_Theta, *Theta])
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


def run_tests():
    test_Y_vs_Y(Ylm, sympy_precision_Ylm, tol=1e-8, l_max=20, use_problem_angles=True)
    test_Y_vs_Y(
        sciYlm, sympy_precision_Ylm, tol=1e-8, l_max=20, use_problem_angles=True
    )

    test_Y_vs_Y(Ylm, jit_Ylm, tol=1e-8, l_max=100, use_problem_angles=True)
    test_Y_vs_Y(real_Ylm, jit_real_Ylm, tol=1e-8, l_max=100, use_problem_angles=True)
    test_Y_vs_Y(Ylm, sciYlm, tol=1e-8, l_max=100, use_problem_angles=True)
    test_Y_vs_Y(real_Ylm, real_sciYlm, tol=1e-8, l_max=100, use_problem_angles=True)
    test_all_Y_vs_Y(
        compute_all_Ylm,
        compute_all_sciYlm,
        tol=1e-8,
        l_max=100,
        use_problem_angles=True,
    )
    test_all_Y_vs_Y(
        compute_all_real_Ylm,
        compute_all_real_sciYlm,
        tol=1e-8,
        l_max=100,
        use_problem_angles=True,
    )
