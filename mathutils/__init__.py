"""MathUtils - Fast mathematical utilities."""

try:
    from .mathutils_backend import *
except ImportError as e:
    raise ImportError(f"Failed to import C++ backend: {e}")

__version__ = "0.1.0"
