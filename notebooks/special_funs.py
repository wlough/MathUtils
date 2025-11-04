# %%
# Legendre polynomials
import sympy as sp
from sympy import legendre


x, t = sp.symbols("x t", real=True)
l_max = 10
range_l = range(0, l_max + 1)
sympy_legendre_polynomials = sp.Array([legendre(_, x) for _ in range_l])
gfun_legendre_polynomials = sp.series(
    1 / sp.sqrt(1 - 2 * x * t + t**2), x=t, n=l_max + 1
)
legendre_polynomials = sp.Array(
    [1, *[gfun_legendre_polynomials.coeff(t**_) for _ in range_l[1:]]]
)


# %%
# Associated Legendre polynomials (Condon–Shortley phase)
import sympy as sp
from sympy import legendre, assoc_legendre


x, t = sp.symbols("x t", real=True)
k, m, n = sp.symbols("k m n", integer=True)
l = sp.symbols(r"\ell", integer=True, nonnegative=True)
l_max = 10
range_l = range(0, l_max + 1)


def get_Plpm_from_Pl(Pl, m):
    return (-1) ** m * sp.sqrt(1 - x**2) ** m * Pl.diff(x, m)


def get_Plmm_from_Plpm(Plpm, l, m):
    return (-1) ** m * Plpm * sp.factorial(l - m) / sp.factorial(l + m)


def get_Plm(l, m):
    abs_m = abs(m)
    if l < 0 or l < abs_m:
        return 0
    Pl = legendre_polynomials[l]
    Plm = get_Plpm_from_Pl(Pl, abs_m)
    if m < 0:
        Plm *= (-1) ** abs_m * sp.factorial(l - abs_m) / sp.factorial(l + abs_m)
    return Plm


P0, P1, P2, P3 = legendre_polynomials[:4]
P10 = P1
P11 = -sp.sqrt(1 - x**2) * P1.diff(x)
P1m1 = -P11 / 2

P20 = P2
P21 = get_Plpm_from_Pl(P2, 1)
P22 = get_Plpm_from_Pl(P2, 2)

# %%
get_Plm(0, 0) - assoc_legendre(0, 0,x)
get_Plm(1, -1) - assoc_legendre(1, -1,x)
get_Plm(1, 0) - assoc_legendre(1, 0,x)
get_Plm(1, 1) - assoc_legendre(1, 1,x)
get_Plm(2, -2) - assoc_legendre(2, -2,x)
get_Plm(2, -1) - assoc_legendre(2, -1,x)
get_Plm(2, 0) - assoc_legendre(2, 0,x)
get_Plm(2, 1) - assoc_legendre(2, 1,x)
get_Plm(2, 2) - assoc_legendre(2, 2,x)


for l in range(0, 3):
    for m in range(-l, l+1):
        print(10*"-")
        print(f"l={l}, m={m}")
        print(f"Plm = {get_Plm(l, m)}")
        print(10*"-")
