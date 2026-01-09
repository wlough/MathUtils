import numpy as np
from numba import njit
from .pyhalf_edge_mesh import *
from .pyhalf_edge_patch import *
from .pyspherical_harmonic_surface import *


@njit
def vf_orientation_correct(a, b, c, cm):
    abc = (a + b + c) / 3
    n = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    return np.dot(n, abc - cm) > 0


@njit
def check_vf_list_orientation(V, F0):
    cm = np.sum(V, axis=0) / len(V)
    F = np.zeros_like(F0)
    for f in range(len(F0)):
        i, j, k = F0[f]
        a, b, c = V[i], V[j], V[k]

        if vf_orientation_correct(a, b, c, cm):
            F[f, :] = np.array([i, j, k])
        else:
            F[f, :] = np.array([i, k, j])
    return F
