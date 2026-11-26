cmake_minimum_required(VERSION 3.18...3.27)
project(SLFlood LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if (NOT SKBUILD)
    message(WARNING "\
    This CMake file is meant to be executed using 'scikit-build-core'.
    Running it directly will almost certainly not produce the desired
    result. If you are a user trying to install this package, use the
    command below, which will install all necessary build dependencies,
    compile the package in an isolated environment, and then install it.
    =====================================================================
    $ pip install .")
endif()

set(DEV_MODULE Development.Module)

find_package(Python 3.10
    REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
    OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    # Name of the extension
    slflood_impl

    # Target the stable ABI for Python 3.12+, which reduces
    # the number of binary wheels that must be built. This
    # does nothing on older Python versions
    STABLE_ABI

    # Optimize for speed instead of size
    NOMINSIZE

    # Source code goes here
    slflood/src/delaunay.cpp
    slflood/src/delaunay_interface.cpp
)

find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)

include(CGAL_Eigen3_support)
target_link_libraries (slflood_impl PRIVATE CGAL::CGAL CGAL::Eigen3_support)

if (SKBUILD)
    install(TARGETS slflood_impl LIBRARY DESTINATION slflood)
endif()
