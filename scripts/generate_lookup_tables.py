#!./.venv/mathutils_test/bin/python3

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from src.python.codegen import (
        write_log_factorial_lookup_table,
        write_spherical_harmonic_index_lookup_table,
    )
    write_log_factorial_lookup_table(n_max=200, precision=50)
    write_spherical_harmonic_index_lookup_table(l_max=200)

