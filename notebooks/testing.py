# from decimal import Decimal, getcontext

# from mathutils.lookup_tables import LOG_FACTORIAL_LOOKUP_TABLE as table_hp

# from src.python.codegen import write_lookup_tables


# write_lookup_tables()
import numpy as np
from mathutils import log_factorial, spherical_harmonic_index_n_LM, Ylm_vectorized
from scipy.special import sph_harm_y

theta = np.pi * np.linspace(0, 1, 11)
phi = np.pi * np.linspace(0, 2, 22)
theta_phi = np.array([[th, ph] for th in theta for ph in phi])
theta, phi = theta_phi.T

l = 11
m = 2
Y = Ylm_vectorized(l, m, theta, phi)
sciY = sph_harm_y(l, m, theta, phi)
Y / sciY
np.linalg.norm(Y - sciY)


def test_Ylm_against_scipy():
    from scipy.special import sph_harm_y

    Ylm_fun = Ylm_vectorized  # up to l=54
    # Ylm_fun = Ylm_alt0  # up to l=10
    # Ylm_fun = Ylm_alt1  # up to l=19

    def n_LM(l, m):
        return m + l**2 + l

    def lm_N(n):
        l = int(np.sqrt(n))
        m = n - (l**2 + l)
        return l, m

    local_tol = 1e-8
    l_min = 0
    l_max = 200
    Theta = np.pi * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    Phi = np.pi * np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    Theta = np.array([*Theta, *np.linspace(0, np.pi, 11)[1:-1]])
    Phi = np.array([*Phi, *np.linspace(2 * np.pi, 22)[1:]])

    ThetaPhi = np.array([[t, p] for t in Theta for p in Phi])
    n_max = n_LM(l_max, l_max)
    n_min = n_LM(l_min, -l_min)
    deltainf = np.zeros(n_max + 1 - n_min)
    LM = np.zeros((n_max + 1 - n_min, 2), dtype=np.int64)
    print(20 * "_")
    l_good = -99999
    m_good = -99999
    for n in range(n_min, n_max + 1):
        _n = n - n_min
        l, m = lm_N(n)
        LM[_n] = l, m
        print(f"testing l={l}, m={m}        ", end="\r")
        Y = Ylm_fun(l, m, *ThetaPhi.T)
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
            m_good} (first {n_LM(l_good, m_good)+1} spherical harmonics)"
    )
    # LMbad = LM[np.where(deltainf > local_tol)]
    # return (deltainf, LMbad)


test_Ylm_against_scipy()
