"""MathUtils - Fast mathematical utilities."""

try:
    from importlib.metadata import version, PackageNotFoundError  # py>=3.8
except Exception:  # very old Pythons only
    from importlib_metadata import version, PackageNotFoundError  # backport

try:
    __version__ = version("pymathutils")
except PackageNotFoundError:
    # Not installed (e.g., running from a source checkout without `pip install -e .`)
    __version__ = "0+unknown"

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
