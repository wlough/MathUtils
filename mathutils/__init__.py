"""MathUtils - Fast mathematical utilities."""

try:
    from .mathutils_backend import *
except ImportError as e:
    raise ImportError(f"Failed to import C++ backend: {e}")

# import stuff in src/python/jit_funs.py to use like jit_funs.stuff
import src.python.jit_funs as jit_funs

__version__ = "0.1.0"
