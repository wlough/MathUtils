# %%
# generate lookup tables
from src.python.codegen import (
    write_log_factorial_lookup_table,
    write_spherical_harmonic_index_lookup_table,
)

# write_lookup_tables(n_max=50, precision=50)
write_log_factorial_lookup_table(n_max=200, precision=200)
write_spherical_harmonic_index_lookup_table(l_max=100)
# then pip install -e .
# import numpy as np
# from math import lgamma

# %%
from mathutils import (
    spherical_harmonic_index_n_LM,
    spherical_harmonic_index_lm_N,
    compute_all_real_Ylm,
)
from mathutils.jit_funs import real_Ylm as jit_real_Ylm
from mathutils import jit_funs
import numpy as np


local_tol = 1e-12
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
# %%
# test sh funs


def test_all_real_Ylm_against_jit():
    import numpy as np
    from mathutils import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        compute_all_real_Ylm,
        jit_funs,
    )

    jit_real_Ylm = jit_funs.real_Ylm
    # Ylm_fun = real_Ylm_vectorized  # up to l=54
    # Ylm_fun = Ylm_alt0  # up to l=10
    # Ylm_fun = Ylm_alt1  # up to l=19
    # np.linspace(0, 1, 12)
    local_tol = 1e-12
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


def test_Ylm_against_scipy():
    import numpy as np
    from scipy.special import sph_harm_y
    from mathutils import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        Ylm_vectorized,
        # real_Ylm_vectorized,
    )

    Ylm_fun = Ylm_vectorized  # up to l=54
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
            m_good} (first {spherical_harmonic_index_n_LM(l_good, m_good)+1} spherical harmonics)"
    )
    # LMbad = LM[np.where(deltainf > local_tol)]
    # return (deltainf, LMbad)


test_Ylm_against_scipy()


# %%


def test_real_Ylm_against_jit():
    import numpy as np
    from src.python.jit_funs import real_Ylm as jit_real_Ylm
    from mathutils import (
        spherical_harmonic_index_n_LM,
        spherical_harmonic_index_lm_N,
        Ylm_vectorized,
        real_Ylm_vectorized,
    )

    # Ylm_fun = real_Ylm_vectorized  # up to l=54
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
        Y = real_Ylm_vectorized(l, m, *ThetaPhi.T)
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


test_real_Ylm_against_jit()


# %%
