import numpy as np
from numba import njit


@njit
def jitnorm(V, axis=-1):
    return np.sqrt(np.sum(V**2, axis=axis))


@njit
def fib_disc(Npoints=100):
    xy = np.zeros((Npoints, 2))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        rad = np.sqrt(i / (Npoints - 1))  # radius at z

        theta = ga * i  # angle increment
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xy[i] = np.array([x, y])

    return xy


@njit
def fib_sphere(Npoints=100):
    xyz = np.zeros((Npoints, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        z = 1.0 - 2 * i / (Npoints - 1)  # -1<=z<=1
        rad = np.sqrt(1.0 - z**2)  # radius at z

        theta = ga * i  # angle increment

        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xyz[i] = np.array([x, y, z])

    return xyz
