from src.python.spherical_harmonics_tests import run_mathutils_vs_scipy_tests


run_mathutils_vs_scipy_tests()


# %%
import numpy as np
from pymathutils.jit_funs import real_Ylm as jit_real_Ylm
from pymathutils.special import (
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

local_tol = 1e-8
l_min = 0
l_max = 100
Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
# Theta = []
# Phi = []
Theta = np.array([*Theta, *np.linspace(0, np.pi, 12)[1:-1]])
Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
minus_one_to_int_pow(2)
ReLogRe_ImLogRe_over_pi(-1.0)

real_Ylm(0, 0, 0, 0)
real_Ylm(0, 0, np.pi / 2, 0.1)

real_Ylm(1, -1, 0, 0)
real_Ylm(1, -1, np.pi / 2, 0)
real_Ylm(1, 0, 0, 0)
real_Ylm(1, 0, np.pi / 2, 0)
real_Ylm(1, 1, 0, 0)
real_Ylm(1, 1, np.pi / 2, 0)

real_Ylm(2, -2, 0, 0)
real_Ylm(2, -2, np.pi / 2, 0)
real_Ylm(2, 0, 0, 0)
real_Ylm(2, 0, np.pi / 2, 0)
real_Ylm(2, 1, 0, 0)
real_Ylm(2, 1, np.pi / 2, 0)
real_Ylm(2, 2, 0, 0)
real_Ylm(2, 2, np.pi / 2, 0)

real_Ylm(22, 0, 0, 0)

Ylm(0, 0, np.pi / 2, 0)
real_Ylm(2, 0, np.pi / 2, 0.0)


sph_harm_y(1, 0, np.pi / 2, 0.0)
real_Ylm(1, 0, np.pi / 2, 0.0)


allY = old_compute_all_real_Ylm(l_max, ThetaPhi)
allY = compute_all_real_Ylm(l_max, ThetaPhi)
allY = compute_all_Ylm(l_max, ThetaPhi)
_allsciY = sph_harm_y_all(l_max, l_max, *ThetaPhi.T)
allsciY = np.zeros_like(allY)
for n in range(spherical_harmonic_index_n_LM(l_max, l_max) + 1):
    l, m = spherical_harmonic_index_lm_N(n)
    allsciY[:, n] = _allsciY[l, m, :]


n_max = spherical_harmonic_index_n_LM(l_max, l_max)
n_min = spherical_harmonic_index_n_LM(l_min, -l_min)
deltainf = np.zeros(n_max + 1 - n_min)
LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
print(20 * "_")
l_good = -99999
m_good = -99999
for n in range(n_min, n_max + 1):
    _n = n - n_min
    l, m = spherical_harmonic_index_lm_N(n)
    LM[_n] = l, m
    print(f"testing l={l}, m={m}        ", end="\r")
    # Y = real_Ylm_vectorized(l, m, *ThetaPhi.T)
    Y = allY[:, n]
    sciY = jit_real_Ylm(l, m, *ThetaPhi.T)
    dY = np.abs((Y - sciY))
    deltainf[_n] = np.linalg.norm(dY, np.inf)
    if deltainf[_n] > local_tol:
        print(f"\nmax error={deltainf[_n]}")
        badThetaPhi = ThetaPhi[np.where(dY > local_tol)] / np.pi
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


# %%
def check_sph_harm_coeff_magnitudes():
    from pymathutils.jit_funs import log_factorial as jit_log_factorial
    from pymathutils.special import (
        log_factorial,
        spherical_harmonic_index_lm_N,
        spherical_harmonic_index_n_LM,
    )
    import numpy as np

    def sph_harm_NlmMklm(k, l, m):
        return np.exp(
            -0.5 * np.log(4 * np.pi)
            - (np.abs(m) + 2 * k) * np.log(2)
            + 0.5 * np.log(2 * l + 1)
            + 0.5 * log_factorial(l + np.abs(m))
            + 0.5 * log_factorial(l - np.abs(m))
            - log_factorial(l - np.abs(m) - 2 * k)
            - log_factorial(np.abs(m) + k)
            - log_factorial(k)
        )

    def sph_harm_Nlm(k, l, m):
        return np.exp(
            -0.5 * np.log(4 * np.pi)
            - (np.abs(m) + 2 * k) * np.log(2)
            + 0.5 * np.log(2 * l + 1)
            + 0.5 * log_factorial(l + np.abs(m))
            + 0.5 * log_factorial(l - np.abs(m))
            - log_factorial(l - np.abs(m) - 2 * k)
            - log_factorial(np.abs(m) + k)
            - log_factorial(k)
        )

    vals_Nklm = []
    klm_indices_Nklm = []
    for l in range(0, 200):
        print(f"l={l}")
        for m in range(-l, l + 1):
            for k in range(0, (l - abs(m)) // 2 + 1):
                klm_indices_Nklm.append([k, l, m])
                vals_Nklm.append(sph_harm_NlmMklm(k, l, m))
        maxNklm = np.max(vals_Nklm)
        klm_maxNklm = klm_indices_Nklm[vals_Nklm.index(maxNklm)]
        print(f"max(Nklm)={maxNklm} at [k,l,m]={klm_maxNklm}")

    maxNklm = np.max(vals_Nklm)
    klm_maxNklm = klm_indices_Nklm[vals_Nklm.index(maxNklm)]

    minNklm = np.min(vals_Nklm)
    klm_minNklm = klm_indices_Nklm[vals_Nklm.index(minNklm)]

    print(f"min(Nklm)={minNklm} at [k,l,m]={klm_minNklm}")
    print(f"max(Nklm)={maxNklm} at [k,l,m]={klm_maxNklm}")


# %%
# test sh funs


def test_all_real_Ylm_against_jit():
    import numpy as np
    from pymathutils import jit_funs
    from pymathutils.special import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        compute_all_real_Ylm,
    )

    jit_real_Ylm = jit_funs.real_Ylm
    # Ylm_fun = real_Ylm_vectorized  # up to l=54
    # Ylm_fun = Ylm_alt0  # up to l=10
    # Ylm_fun = Ylm_alt1  # up to l=19
    # np.linspace(0, 1, 12)
    local_tol = 1e-8
    l_min = 0
    l_max = 100
    Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    # Theta = []
    # Phi = []
    Theta = np.array([*Theta, *np.linspace(0, np.pi, 12)[1:-1]])
    Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    allY = compute_all_real_Ylm(l_max, ThetaPhi)
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)
    n_min = spherical_harmonic_index_n_LM(l_min, -l_min)
    deltainf = np.zeros(n_max + 1 - n_min)
    LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
    print(20 * "_")
    l_good = -99999
    m_good = -99999
    for n in range(n_min, n_max + 1):
        _n = n - n_min
        l, m = spherical_harmonic_index_lm_N(n)
        LM[_n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        # Y = real_Ylm_vectorized(l, m, *ThetaPhi.T)
        Y = allY[:, n]
        sciY = jit_real_Ylm(l, m, *ThetaPhi.T)
        dY = np.abs((Y - sciY))
        deltainf[_n] = np.linalg.norm(dY, np.inf)
        if deltainf[_n] > local_tol:
            print(f"\nmax error={deltainf[_n]}")
            badThetaPhi = ThetaPhi[np.where(dY > local_tol)] / np.pi
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
    # LMbad = LM[np.where(deltainf > local_tol)]
    # return (deltainf, LMbad)


test_all_real_Ylm_against_jit()

# %%


def get_allYlm_scipy(l_max, ThetaPhi):
    from scipy.special import sph_harm_y_all


def test_all_Ylm_against_scipy():
    import numpy as np
    from scipy.special import sph_harm_y
    from pymathutils.special import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        compute_all_Ylm,
    )

    # Ylm_fun = Ylm_vectorized  # up to l=54
    # Ylm_fun = Ylm_alt0  # up to l=10
    # Ylm_fun = Ylm_alt1  # up to l=19

    local_tol = 1e-8
    l_min = 0
    l_max = 200
    Theta = []
    Phi = []
    # Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25,
    #                        0.0, 0.25, 0.5, 0.75, 1.0])
    Theta = np.array([*Theta, *np.linspace(0, np.pi, 11)[1:-1]])
    Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    allY = compute_all_Ylm(l_max, ThetaPhi)
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)
    n_min = spherical_harmonic_index_n_LM(l_min, -l_min)
    deltainf = np.zeros(n_max + 1 - n_min)
    LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
    print(20 * "_")
    l_good = -99999
    m_good = -99999
    for n in range(n_min, n_max + 1):
        _n = n - n_min
        l, m = spherical_harmonic_index_lm_N(n)
        LM[_n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y = allY[:, n]
        sciY = sph_harm_y(l, m, *ThetaPhi.T)
        dY = np.abs((Y - sciY))
        deltainf[_n] = np.linalg.norm(dY, np.inf)
        if deltainf[_n] > local_tol:
            print(f"\nmax error={deltainf[_n]}")
            badThetaPhi = ThetaPhi[np.where(dY > local_tol)] / np.pi
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
    # LMbad = LM[np.where(deltainf > local_tol)]
    # return (deltainf, LMbad)


test_all_Ylm_against_scipy()


# %%


def test_Ylm_against_scipy():
    import numpy as np
    from scipy.special import sph_harm_y
    from pymathutils.special import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        Ylm,
        # real_Ylm_vectorized,
    )

    Ylm_fun = Ylm  # up to l=54
    # Ylm_fun = Ylm_alt0  # up to l=10
    # Ylm_fun = Ylm_alt1  # up to l=19

    local_tol = 1e-8
    l_min = 0
    l_max = 200
    Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    Theta = np.array([*Theta, *np.linspace(0, np.pi, 11)[1:-1]])
    Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)
    n_min = spherical_harmonic_index_n_LM(l_min, -l_min)
    deltainf = np.zeros(n_max + 1 - n_min)
    LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
    print(20 * "_")
    l_good = -99999
    m_good = -99999
    for n in range(n_min, n_max + 1):
        _n = n - n_min
        l, m = spherical_harmonic_index_lm_N(n)
        LM[_n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y = Ylm_fun(l, m, ThetaPhi)
        sciY = sph_harm_y(l, m, *ThetaPhi.T)
        dY = np.abs((Y - sciY))
        deltainf[_n] = np.linalg.norm(dY, np.inf)
        if deltainf[_n] > local_tol:
            print(f"\nmax error={deltainf[_n]}")
            badThetaPhi = ThetaPhi[np.where(dY > local_tol)] / np.pi
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
    # LMbad = LM[np.where(deltainf > local_tol)]
    # return (deltainf, LMbad)


test_Ylm_against_scipy()


# %%
def test_scipy_Ylm_against_sympy():
    import sympy as sp
    import numpy as np
    from pymathutils.special import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
    )
    from scipy.special import sph_harm_y

    print("----------------------------")
    print("test_scipy_Ylm_against_sympy")

    def sympy_reference(l, m, theta_val, phi_val, precision=50):
        """Simplified SymPy reference with proper precision handling"""

        # Create symbolic variables
        theta, phi = sp.symbols("theta phi", real=True)

        # Get symbolic expression
        ylm_symbolic = sp.Ynm(l, m, theta, phi)

        # Substitute values and evaluate with specified precision
        result = ylm_symbolic.subs(
            {theta: sp.Float(theta_val, precision), phi: sp.Float(phi_val, precision)}
        ).evalf(precision)

        # Handle complex results
        if result.is_real:
            return complex(float(result), 0.0)
        else:
            real_part = float(sp.re(result))
            imag_part = float(sp.im(result))
            return complex(real_part, imag_part)

    def compare_with_sympy(your_function, l, m, theta_vals, phi_vals):
        """Compare your results with SymPy reference"""

        errors = []

        for theta, phi in zip(theta_vals, phi_vals):
            # Your result
            your_result = your_function(l, m, theta, phi)

            # SymPy reference
            sympy_result = sympy_reference(l, m, theta, phi)

            # Compute error
            abs_error = abs(your_result - sympy_result)
            rel_error = (
                abs_error / abs(sympy_result) if sympy_result != 0 else abs_error
            )

            errors.append(
                {
                    "theta": theta,
                    "phi": phi,
                    "your_result": your_result,
                    "sympy_result": sympy_result,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                }
            )

        return errors

    local_tol = 1e-12
    l_min = 10
    l_max = 100
    Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    Theta = np.array([*Theta, *np.linspace(0, np.pi, 11)[1:-1]])
    Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)
    n_min = spherical_harmonic_index_n_LM(l_min, -l_min)
    deltainf = np.zeros(n_max + 1 - n_min)
    LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
    print(20 * "_")
    l_good = -99999
    m_good = -99999
    for n in range(n_min, n_max + 1):
        _n = n - n_min
        l, m = spherical_harmonic_index_lm_N(n)
        LM[_n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y = np.array([sph_harm_y(l, m, th, ph) for th, ph in ThetaPhi])
        symY = np.array([sympy_reference(l, m, th, ph) for th, ph in ThetaPhi])
        dY = np.abs((Y - symY))
        deltainf[_n] = np.linalg.norm(dY, np.inf)
        if deltainf[_n] > local_tol:
            print(f"\nmax error={deltainf[_n]}")
            badThetaPhi = ThetaPhi[np.where(dY > local_tol)] / np.pi
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


test_scipy_Ylm_against_sympy()
# %%


def test_mathutils_Ylm_against_sympy():
    import sympy as sp
    import numpy as np
    from pymathutils.special import (
        Ylm,
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
    )

    print("--------------------------------")
    print("test_mathutils_Ylm_against_sympy")

    def sympy_reference(l, m, theta_val, phi_val, precision=50):
        """Simplified SymPy reference with proper precision handling"""

        # Create symbolic variables
        theta, phi = sp.symbols("theta phi", real=True)

        # Get symbolic expression
        ylm_symbolic = sp.Ynm(l, m, theta, phi)

        # Substitute values and evaluate with specified precision
        result = ylm_symbolic.subs(
            {theta: sp.Float(theta_val, precision), phi: sp.Float(phi_val, precision)}
        ).evalf(precision)

        # Handle complex results
        if result.is_real:
            return complex(float(result), 0.0)
        else:
            real_part = float(sp.re(result))
            imag_part = float(sp.im(result))
            return complex(real_part, imag_part)

    def compare_with_sympy(your_function, l, m, theta_vals, phi_vals):
        """Compare your results with SymPy reference"""

        errors = []

        for theta, phi in zip(theta_vals, phi_vals):
            # Your result
            your_result = your_function(l, m, theta, phi)

            # SymPy reference
            sympy_result = sympy_reference(l, m, theta, phi)

            # Compute error
            abs_error = abs(your_result - sympy_result)
            rel_error = (
                abs_error / abs(sympy_result) if sympy_result != 0 else abs_error
            )

            errors.append(
                {
                    "theta": theta,
                    "phi": phi,
                    "your_result": your_result,
                    "sympy_result": sympy_result,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                }
            )

        return errors

    local_tol = 1e-12
    l_min = 0
    l_max = 100
    Theta = np.pi * np.array([0.0, 0.5, 1.0])
    Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    # Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25,
    #                        0.0, 0.25, 0.5, 0.75, 1.0])
    Theta = np.array([*Theta, *np.linspace(0, np.pi, 11)[1:-1]])
    Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)
    n_min = spherical_harmonic_index_n_LM(l_min, -l_min)
    deltainf = np.zeros(n_max + 1 - n_min)
    LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
    print(20 * "_")
    l_good = -99999
    m_good = -99999
    for n in range(n_min, n_max + 1):
        _n = n - n_min
        l, m = spherical_harmonic_index_lm_N(n)
        LM[_n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y = np.array([Ylm(l, m, th, ph) for th, ph in ThetaPhi])
        symY = np.array([sympy_reference(l, m, th, ph) for th, ph in ThetaPhi])
        dY = np.abs((Y - symY))
        deltainf[_n] = np.linalg.norm(dY, np.inf)
        if deltainf[_n] > local_tol:
            print(f"\nmax error={deltainf[_n]}")
            badThetaPhi = ThetaPhi[np.where(dY > local_tol)] / np.pi
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


test_mathutils_Ylm_against_sympy()
