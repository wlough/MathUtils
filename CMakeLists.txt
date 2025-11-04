cmake_minimum_required(VERSION 3.18)
project(pymathutils LANGUAGES CXX)

# Python & pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Users can override with -DEigen3_DIR=... or EIGEN3_INCLUDE_DIR if needed.
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Eigen3 REQUIRED NO_MODULE)

# extension target
pybind11_add_module(pymathutils_backend
    src/cpp/bindings.cpp
    src/cpp/findiff.cpp
)

target_include_directories(pymathutils_backend
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pymathutils_backend PRIVATE cxx_std_20)
target_link_libraries(pymathutils_backend PRIVATE Eigen3::Eigen)

# Python module "mathutils_backend"
set_target_properties(pymathutils_backend PROPERTIES OUTPUT_NAME "mathutils_backend")

# Put the built extension inside the pymathutils package in the wheel
# SKBUILD_PLATLIB_DIR is provided by scikit-build-core
install(TARGETS pymathutils_backend
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/pymathutils"
        ARCHIVE DESTINATION "${SKBUILD_PLATLIB_DIR}/pymathutils"
        RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/pymathutils")

# ship headers for downstream C++ users (not required for Python runtime)
# install(DIRECTORY include/ DESTINATION include)