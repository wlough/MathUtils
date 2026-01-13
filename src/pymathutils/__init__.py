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


# general utility functions
from .mathutils_backend import (
    sign,
    thetaphi_from_xyz,
    rthetaphi_from_xyz,
    xyz_from_rthetaphi,
)

from . import pyutils


# mesh utilities
from . import mesh

# special functions
from . import special

# finite difference utilities
from . import findiff

from . import random


__all__ = [
    "special",
    "findiff",
    "mesh",
    "sign",
    "thetaphi_from_xyz",
    "rthetaphi_from_xyz",
    "xyz_from_rthetaphi",
]
