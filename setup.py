from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

ext_modules = [
    Pybind11Extension(
        "mathutils.mathutils_backend",
        [
            "src/cpp/bindings.cpp",  # Correct path to your bindings
        ],
        include_dirs=[
            "include",  # Your header files
            "/usr/include/eigen3",  # Eigen headers
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="mathutils",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
