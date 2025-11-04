"""MathUtils - Fast mathematical utilities."""

# numba jit compiled functions
from . import jit_funs

# general utility functions
from .mathutils_backend import (
    sign,
    thetaphi_from_xyz,
    rthetaphi_from_xyz,
    xyz_from_rthetaphi,
)

# special functions
from . import special

# finite difference utilities
from . import findiff


# """MathUtils - Fast mathematical utilities."""

# try:
#     from .mathutils_backend import *
# except ImportError as e:
#     raise ImportError(f"Failed to import C++ backend: {e}")

# # Now this will work since jit_funs.py is in the same directory
# from . import jit_funs

# __version__ = "0.1.0"
