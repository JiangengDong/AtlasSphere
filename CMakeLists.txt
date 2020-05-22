cmake_minimum_required (VERSION 3.10)
project(AtlasSphere)

find_package(ompl 1.4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost 1.58 QUIET REQUIRED COMPONENTS serialization filesystem system program_options chrono)

# add_subdirectory(pybind11)
add_subdirectory(src)